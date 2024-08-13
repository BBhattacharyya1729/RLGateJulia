####Must implement the following
#    - RLBase.act!(env::AbstractEnv, a::Vector{Float32})
#    - RLBase.reward(env::AbstractEnv,action::Union{Vector{Float32},Nothing};kwargs...) -> Float32
###
#    Assumes actions are normalized (-1,1)^n

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


###Utilities 
function log_std_clip(x)
    return minimum([-4,maximum([x,-4])])
end
clip(x,ϵ) = minimum([1+ϵ,maximum([x,1-ϵ])]) 
g(ϵ,A) = (A>=0) ? (1+ϵ) * A : (1-ϵ) * A


###Actor Critic (SDE style)
struct ActorCriticPolicy
    feature_network::Chain
    mean_network::Dense
    std_network::Dense
    value_network::Chain
end


###Apply policy to env
function(𝒫::ActorCriticPolicy)(env::AbstractEnv; deterministic::Bool = false)
    state = Vector{Float32}(RLBase.state(env))
    means = 𝒫.mean_network(𝒫.feature_network(state))
    if(!deterministic)
        std = exp(𝒫.std_network(𝒫.feature_network(state))[1])
        return means+rand(Normal(0,std),length(means))
    else
        return means
    end        
end

###Construct appropriate networks
function ActorCriticPolicy(env::AbstractEnv;l::Vector{Int64}=[64,64])
    out_size = DomainSets.dimension(RLBase.action_space(env))
    in_size = DomainSets.dimension(RLBase.state_space(env))

    feature_network = Chain(Dense(in_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in 1:length(l)-1]...)
    mean_network = Dense(l[end]=>out_size,tanh)
    std_network = Dense(l[end]=>1,log_std_clip)
    value_network =  Chain(Dense(in_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in 1:length(l)-1]...,Dense(l[end]=>1))
    
    return ActorCriticPolicy(feature_network,mean_network,std_network,value_network)
end

Flux.@functor ActorCriticPolicy


###log probability
function policy_log_prob(𝒫::ActorCriticPolicy,state::Vector{Float32},action::Vector{Float32})
    means = 𝒫.mean_network(𝒫.feature_network(state))
    std = exp(𝒫.std_network(𝒫.feature_network(state))[1])
    devs = action-means       
    return Float32(sum((-devs.^2/(2*std^2)).-1/2 * log(2 * pi * std^2)))
end

###Sample trajectories
function sample_trajectory(𝒫::ActorCriticPolicy,env::AbstractEnv ;deterministic::Bool=false, kwargs...)
    RLBase.reset!(env)
    rewards = Vector{Float32}()
    actions = Vector{Vector{Float32}}()
    states  = Vector{Vector{Float64}}()
    while(! RLBase.is_terminated(env))
        push!(states,deepcopy(RLBase.state(env)))
        action = 𝒫(env;deterministic=deterministic)
        push!(actions,action)
        push!(rewards,deepcopy(RLBase.reward(env,action;kwargs...)))
        RLBase.act!(env,action) 
    end
    rewards[end]+= RLBase.reward(env;kwargs...)
    return rewards,actions,Vector{Vector{Float32}}(states)
end


###Advantage Estimation
discount_cumsum(l::Vector{Float32},γ::Float32) = [sum(l[j:end].*[γ^i for i in 0:length(l)-j]) for j in 1:length(l)]

function GAE(states::Vector{Vector{Float32}},rewards::Vector{Float32},actions::Vector{Vector{Float32}},𝒫::ActorCriticPolicy;γ::Float32=Float32(0.99),λ::Float32=Float32(0.97))
    vals = getindex.(𝒫.value_network.(states),1)
    push!(vals,vals[end])
    δ = rewards[1:end] + γ * vals[2:end] - vals[1:end-1]
    return discount_cumsum(rewards,γ),discount_cumsum(δ,γ*λ)
end


###Surrogate Loss
function actor_critic_Loss(
                𝒫::ActorCriticPolicy,
                rewards_to_go::Vector{Float32},
                states_list::Matrix{Float32},
                acts_list::Matrix{Float32},
                Adv_list::Vector{Float32},
                old_log_probs::Vector{Float32};
                ϵ::Float32 = 2f-1,
                norm_adv::Bool=true
               )
    if(norm_adv)
        Adv_list = (Adv_list.-mean(Adv_list))/std(Adv_list)
    end
    value_loss = mean(abs2.(𝒫.value_network(states_list)'-rewards_to_go))
    new_log_probs = [policy_log_prob(𝒫,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)]
    ratio = exp.(new_log_probs.-old_log_probs)
    clip_ratio = mean(ratio.*Adv_list.>clip.(ratio, ϵ).*Adv_list)
    policy_loss =  mean(minimum([ratio.*Adv_list clip.(ratio, ϵ).*Adv_list],dims=2))
    entropy  =  mean(-new_log_probs)
    return  policy_loss,value_loss,entropy,clip_ratio
end

###PPO
function PPO(env::AbstractEnv;
             trajectory_batch_size::Int64=20,
             n_steps::Int64 =2048,
             iterations::Int64=100,
             initial_policy::Union{ActorCriticPolicy,Nothing}=nothing,
             l::Vector{Int64}=[64,64],
             epochs::Int64 = 10, 
             η::Float32 = 1f-3,
             fit_batch_size::Int64=64,
             ϵ::Float32 = 1f-1,
             verbose::Bool=true,
             γ::Float32=Float32(0.99),
             λ::Float32=Float32(0.97),
             KL_targ::Union{Float32,Nothing}=5f-2,
             clip_grad_tresh::Float32 = 5f-1,
             vf_ratio::Float32 = 5f-1,
             ent_ratio::Float32 = 0f0,
             norm_adv::Bool = true,
             trajectory_kwargs...)

    𝒫 = isnothing(initial_policy) ? ActorCriticPolicy(env;l=l) : initial_policy
    opt_state = Flux.setup(Flux.Optimiser(ClipValue(clip_grad_tresh), ADAM(η)), 𝒫)
    n = DomainSets.dimension(RLBase.action_space(env))
    score_history = Vector{Float32}()
    for iter in 1:iterations
        rewards_to_go = Vector{Vector{Float32}}()
        Adv_list =   Vector{Vector{Float32}}()
        acts_list = Vector{Vector{Float32}}()
        states_list = Vector{Vector{Float32}}()
        scores = Vector{Float32}()
        while(length(rewards_to_go) < n_steps || length(scores)<trajectory_batch_size)
            rewards,acts,states = sample_trajectory(𝒫,env;trajectory_kwargs...)
            rtg,adv = GAE(states,rewards,acts,𝒫;γ=γ,λ=λ)
            push!(scores,sum(rewards))
            rewards_to_go = vcat(rewards_to_go,rtg)
            Adv_list = vcat(Adv_list,adv)
            
            states_list=vcat(states_list,states)
            acts_list=vcat(acts_list,acts)
        end
        acts_list = Matrix{Float32}([acts_list[j][i] for i=1:size(acts_list[1])[1], j=1:size(acts_list)[1]])
        states_list = Matrix{Float32}([states_list[j][i] for i=1:size(states_list[1])[1], j=1:size(states_list)[1]])
        rewards_to_go =  Matrix{Float32}(reshape(rewards_to_go,1,length(rewards_to_go)))
        
        Adv_list =  Matrix{Float32}(reshape(Adv_list,1,length(Adv_list)))
        

        old_log_probs = reshape([policy_log_prob(𝒫,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
        data = Flux.DataLoader((reduce(vcat,[rewards_to_go,Adv_list,old_log_probs]), vcat(states_list,acts_list)), batchsize=fit_batch_size, shuffle=true, partial=false)

        policy_losses = Vector{Float32}()
        value_losses = Vector{Float32}()
        clip_ratios = Vector{Float32}()
        entropy_losses = Vector{Float32}()
        KL_list = Vector{Float32}()
        for epoch in 1:epochs
            for (x,y) in data

              batch_rewards_to_go=x[1,:]
              batch_Adv=x[2,:]
              batch_old_log_probs=x[3,:]
              batch_states = y[1:end-n,:]
              batch_acts = y[end-n+1:end,:]
                
              p_loss,v_loss,entropy,clip_rat,new_log_probs,ratio =0f0,0f0,0f0,0f0,0f0,0f0
              val, grads = Flux.withgradient(𝒫) do 𝒫
                  p_loss,v_loss,entropy,clip_rat = actor_critic_Loss(
                                    𝒫,
                                    batch_rewards_to_go,
                                    batch_states,
                                    batch_acts,
                                    batch_Adv,
                                    batch_old_log_probs;
                                    ϵ=ϵ,
                                    norm_adv = norm_adv
                                   )

                 return vf_ratio * v_loss - p_loss - ent_ratio * entropy
              end

            push!(policy_losses,p_loss)
            push!(value_losses,v_loss)
            push!(clip_ratios,clip_rat)
            push!(entropy_losses,entropy)

            Flux.update!(opt_state, 𝒫, grads[1])
            
            new_log_probs = reshape([policy_log_prob(𝒫,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
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
            @printf "Total Steps: %i\n" length(rewards_to_go)
            @printf "Total Trajectories: %i\n" length(scores)
            @printf "Final KL: %.5f\n" KL_list[end]
            @printf "Mean Policy Loss: %.5f\n" mean(policy_losses)
            @printf "Mean Value Loss: %.5f\n" mean(value_losses)
            @printf "Mean Entropy: %.5f\n" mean(entropy_losses)
            @printf "Mean Clip Ratio: %.5f\n" mean(clip_ratios)
            println("-------------------------")
            flush(stdout)
        end
        push!(score_history,mean(scores))
    end
    return 𝒫,score_history
end