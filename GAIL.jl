using QuantumCollocation
using NamedTrajectories
using TrajectoryIndexingUtils
using Flux
using ReinforcementLearning
using IntervalSets
using LinearAlgebra
using Base
using Distributions
using Statistics
using Printf
using Reexport
using Revise
using DomainSets
includet("PPO.jl")

struct ActorDiscriminator
    feature_network::Chain
    mean_network::Dense
    std_network::Dense
    discriminator::Chain
end



###Apply policy to env
function(𝒟::ActorDiscriminator)(env::AbstractEnv; deterministic::Bool = false)
    state = Vector{Float32}(RLBase.state(env))
    means = 𝒟.mean_network(𝒟.feature_network(state))
    if(!deterministic)
        std = exp(𝒟.std_network(𝒟.feature_network(state))[1])
        return means+rand(Normal(0,std),length(means))
    else
        return means
    end        
end

###Construct appropriate networks
function ActorDiscriminator(env::AbstractEnv;l::Vector{Int64}=[64,64])
    out_size = DomainSets.dimension(RLBase.action_space(env))
    in_size = DomainSets.dimension(RLBase.state_space(env))

    feature_network = Chain(Dense(in_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in 1:length(l)-1]...)
    mean_network = Dense(l[end]=>out_size,tanh)
    std_network = Dense(l[end]=>1,log_std_clip)
    discriminator =  Chain(Dense(in_size+out_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in 1:length(l)-1]...,Dense(l[end]=>1,sigmoid))
    
    return ActorDiscriminator(feature_network,mean_network,std_network,discriminator)
end

Flux.@functor ActorDiscriminator


###log probability
function policy_log_prob(𝒟::ActorDiscriminator,state::Vector{Float32},action::Vector{Float32})
    means = 𝒟.mean_network(𝒟.feature_network(state))
    std = exp(𝒟.std_network(𝒟.feature_network(state))[1])
    devs = action-means       
    return Float32(sum((-devs.^2/(2*std^2)).-1/2 * log(2 * pi * std^2)))
end

###Sample trajectories
function sample_trajectory(𝒟::ActorDiscriminator,env::AbstractEnv;deterministic::Bool=false, kwargs...)
    RLBase.reset!(env)
    rewards = Vector{Float32}()
    actions = Vector{Vector{Float32}}()
    states  = Vector{Vector{Float64}}()
    while(! RLBase.is_terminated(env))
        push!(states,deepcopy(RLBase.state(env)))
        action = 𝒟(env;deterministic=deterministic)
        push!(actions,action)
        push!(rewards,deepcopy(RLBase.reward(env,action;kwargs...)))
        RLBase.act!(env,action) 
    end
    rewards[end]+= deepcopy(RLBase.reward(env,nothing;kwargs...))
    return rewards,actions,Vector{Vector{Float32}}(states)
end

function discriminator_loss(𝒟::ActorDiscriminator,
                            expert_states::Matrix{Float32},
                            expert_acts::Matrix{Float32},
                            states_list::Matrix{Float32},
                            acts_list::Matrix{Float32}
                           )
    return mean(log.([𝒟.discriminator(vcat(s,a))[1] for (s,a) in zip([states_list[:,i] for i in 1:size(states_list)[end]],[acts_list[:,i] for i in 1:size(acts_list)[end]])]))
    + mean(log.(1 .- [𝒟.discriminator(vcat(s,a))[1] for (s,a) in zip([expert_states[:,i] for i in 1:size(expert_states)[end]],[expert_acts[:,i] for i in 1:size(expert_acts)[end]])]))
end

function policy_loss(𝒟::ActorDiscriminator,
                            expert_states::Matrix{Float32},
                            expert_acts::Matrix{Float32},
                            states_list::Matrix{Float32},
                            acts_list::Matrix{Float32},
                            old_log_probs::Vector{Float32};
                            ϵ::Float32 = 1f-1,
                           )

    l = [𝒟.discriminator(vcat(s,a))[1] for (s,a) in zip([states_list[:,i] for i in 1:size(states_list)[end]],[acts_list[:,i] for i in 1:size(acts_list)[end]])]
    Q = log.(l)
    new_log_probs = [policy_log_prob(𝒟,[s,a]...) for (s,a) in zip([states_list[:,i] for i in 1:size(states_list)[end]],[acts_list[:,i] for i in 1:size(acts_list)[end]])]
    log_ratio=(new_log_probs.-old_log_probs)
    ratio = exp.(log_ratio)

    clip_ratio = mean(ratio.*Q.>clip.(ratio, ϵ).*Q)
    p_loss =  mean(new_log_probs.*Q)
    entropy  =  mean(-new_log_probs)
    
    return p_loss,entropy,clip_ratio
end

function GAIL(env::AbstractEnv,
             expert_states::Matrix{Float32},
             expert_acts::Matrix{Float32};
             iterations::Int64=100,
             initial_policy::Union{ActorDiscriminator,Nothing}=nothing,
             l::Vector{Int64}=[64,64],
             epochs::Int64 = 10, 
             η::Float32 = 1f-3,
             fit_batch_size::Int64=64,
             ϵ::Float32 = 1f-1,
             verbose::Bool=true,
             KL_targ::Union{Float32,Nothing}=5f-2,
             clip_grad_tresh::Union{Float32} = 1f1,
             ent_ratio::Float32 = 0f0,
             use_log_rewards::Bool = false,
             trajectory_kwargs...   
        )
    println("kms!")
    𝒟 = isnothing(initial_policy) ? ActorDiscriminator(env;l=l) : initial_policy
    discriminator_opt = Flux.setup(Flux.Optimiser(ClipValue(clip_grad_tresh), ADAM(η)), 𝒟)
    Flux.freeze!(discriminator_opt.feature_network)
    Flux.freeze!(discriminator_opt.std_network)
    Flux.freeze!(discriminator_opt.mean_network)

    policy_opt = Flux.setup(Flux.Optimiser(ClipValue(clip_grad_tresh), ADAM(η)), 𝒟)
    Flux.freeze!(policy_opt.discriminator)

    for iter in 1:iterations
        scores = Vector{Float32}()
        acts_list = Vector{Vector{Float32}}()
        states_list = Vector{Vector{Float32}}()
        while(length(states_list) < size(expert_states)[end])
            rewards,acts,states = sample_trajectory(𝒟,env;trajectory_kwargs...)
            if(use_log_rewards)
                rewards = dB.(rewards)
            end
            push!(scores,sum(rewards))
            rewards = 
            states_list=vcat(states_list,states)
            acts_list=vcat(acts_list,acts)
        end
        acts_list = Matrix{Float32}([acts_list[j][i] for i=1:size(acts_list[1])[1], j=1:size(acts_list)[1]])
        states_list = Matrix{Float32}([states_list[j][i] for i=1:size(states_list[1])[1], j=1:size(states_list)[1]])

        old_log_probs = [policy_log_prob(𝒟,[s,a]...) for (s,a) in zip([states_list[:,i] for i in 1:size(states_list)[end]],[acts_list[:,i] for i in 1:size(acts_list)[end]])]

        policy_losses = Vector{Float32}()
        discriminator_losses = Vector{Float32}()
        clip_ratios = Vector{Float32}()
        entropy_losses = Vector{Float32}()
        KL_list = Vector{Float32}()
data = Flux.DataLoader((vcat(states_list,old_log_probs'),acts_list), batchsize=fit_batch_size, shuffle=true, partial=false)
        for epoch in 1:epochs
            for (x,y) in data
            batch_states =x[1:end-1,:]
            batch_old_log_probs = x[end,:]
            batch_acts = y
            d_score = 0f0
             val, grads = Flux.withgradient(𝒟) do 𝒟
                d_score = discriminator_loss(𝒟,
                                expert_states,
                                expert_acts,
                                batch_states,
                                batch_acts
                               )
                return -d_score
            end
            Flux.update!(discriminator_opt, 𝒟, grads[1])
            push!(discriminator_losses,d_score)
          end
              end
        for epoch in 1:epochs
            for (x,y) in data
            batch_states =x[1:end-1,:]
            batch_old_log_probs = x[end,:]
            batch_acts = y
            p_loss,entropy,clip_ratio = 0f0,0f0,0f0

            val, grads = Flux.withgradient(𝒟) do 𝒟

              p_loss,entropy,clip_ratio = 
                             policy_loss(𝒟,
                             expert_states,
                             expert_acts,
                             batch_states,
                             batch_acts,
                             batch_old_log_probs;
                             ϵ=ϵ,
                               )
                return p_loss - ent_ratio*entropy
            end
            Flux.update!(policy_opt, 𝒟, grads[1])
            push!(policy_losses,p_loss)
            push!(clip_ratios,clip_ratio)
            push!(entropy_losses,entropy)

            new_log_probs = reshape([policy_log_prob(𝒟,states_list[:,i],acts_list[:,i]) for i in 1:length(old_log_probs)],size(old_log_probs)...)
            log_ratio = new_log_probs.-old_log_probs
            KL = mean((exp.(log_ratio).- 1).-log_ratio)
            push!(KL_list,KL)
            if(!isnothing(KL_targ) && KL>KL_targ)
                break
            end  
  end
              end         
                

        if(verbose)
            @printf "Iterations %i Complete\n" iter
            @printf "Updates %i\n" length(KL_list)
            @printf "Avg Score: %.5f\n" mean(scores)
            @printf "Total Trajectories: %i\n" length(scores)
            @printf "Final KL: %.5f\n" KL_list[end]
            @printf "Mean Policy Loss: %.5f\n" mean(policy_losses)
            @printf "Start Policy Loss: %.5f\n" policy_losses[1]
            @printf "End Policy Loss: %.5f\n" policy_losses[end]
            @printf "Mean Discriminator Loss: %.5f\n" mean(discriminator_losses)
            @printf "Start Discriminator Loss: %.5f\n" discriminator_losses[1]
            @printf "End Discriminator Loss: %.5f\n" discriminator_losses[end]
            @printf "Mean Entropy: %.5f\n" mean(entropy_losses)
            @printf "Mean Clip Ratio: %.5f\n" mean(clip_ratios)
            println("-------------------------")
            flush(stdout)
        end
    end
    return 𝒟,score_history
end