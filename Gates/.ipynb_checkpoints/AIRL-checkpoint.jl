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
    g::Chain
    h::Chain
    γ::Float32
    𝒫::ActorCriticPolicy
end

Flux.@functor ActorDiscriminator


##convert sample trajectory to (s,a,s') and rewards
function sas_sample(𝒫::ActorCriticPolicy,env::AbstractEnv;deterministic::Bool=false, kwargs...)
        rewards,acts,states = sample_trajectory(𝒫,env;deterministic=deterministic, kwargs...)
        push!(states,RLBase.state(env))
        return rewards,acts,Vector{Vector{Float32}}(states[1:end-1]),Vector{Vector{Float32}}(states[2:end])
end

### f function
function f(𝒟::ActorDiscriminator,state::Vector{Float32},action::Vector{Float32},new_state::Vector{Float32})
    return 𝒟.g(vcat(state,action))[1] + 𝒟.γ * 𝒟.h(new_state)[1] - 𝒟.h(state)[1]
end


###Get Discriminator Values
function (𝒟::ActorDiscriminator)(state::Vector{Float32},action::Vector{Float32},new_state::Vector{Float32})
    return exp(f(𝒟,state,action,new_state)[1])/(exp(f(𝒟,state,action,new_state)[1]) + exp(policy_log_prob(𝒟.𝒫,state,action)))
end

function  (𝒟::ActorDiscriminator)(states_list::Vector{Vector{Float32}},acts_list::Vector{Vector{Float32}},new_states_list::Vector{Vector{Float32}})
    return [𝒟(states_list[i],acts_list[i],new_states_list[i]) for i in 1:length(states_list)]
end 


###Construct appropriate networks
function ActorDiscriminator(env::AbstractEnv;γ::Float32=Float32(0.99),l::Vector{Int64}=[128,128])
    out_size = DomainSets.dimension(RLBase.action_space(env))
    in_size = DomainSets.dimension(RLBase.state_space(env))


    g =  Chain(Dense(in_size+out_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in   1:length(l)-1]...,Dense(l[end]=>1))
    h =  Chain(Dense(in_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in   1:length(l)-1]...,Dense(l[end]=>1))
    
    return ActorDiscriminator(g,h,γ,ActorCriticPolicy(env,l=l))
end

###Construct appropriate networks
function ActorDiscriminator(env::AbstractEnv,𝒫::ActorCriticPolicy;γ::Float32=Float32(0.99),l::Vector{Int64}=[128,128])
    out_size = DomainSets.dimension(RLBase.action_space(env))
    in_size = DomainSets.dimension(RLBase.state_space(env))


    g =  Chain(Dense(in_size+out_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in   1:length(l)-1]...,Dense(l[end]=>1))
    h =  Chain(Dense(in_size=>l[1],tanh),[Dense(l[i]=>l[i+1],tanh) for i in   1:length(l)-1]...,Dense(l[end]=>1))
    
    return ActorDiscriminator(g,h,γ,copy(𝒫))
end


function discriminator_loss(𝒟::ActorDiscriminator,
                            expert_states::Vector{Vector{Float32}},
                            expert_acts::Vector{Vector{Float32}},
                            expert_new_states::Vector{Vector{Float32}},
                            states_list::Vector{Vector{Float32}},
                            acts_list::Vector{Vector{Float32}},
                            new_states_list::Vector{Vector{Float32}}
                           )
    return mean(log.(1 .- 𝒟(states_list,acts_list,new_states_list))),mean(log.(𝒟(expert_states,expert_acts,expert_new_states)))
end


###PPO
function AIRL(env::AbstractEnv,
              expert_states::Vector{Vector{Float32}},
              expert_acts::Vector{Vector{Float32}},
              expert_new_states::Vector{Vector{Float32}};
             trajectory_batch_size::Int64=20,
             n_steps::Int64 =2048,
             iterations::Int64=100,
             initial_policy::Union{ActorCriticPolicy,Nothing}=nothing,
             l::Vector{Int64}=[128,128],
             policy_epochs::Int64 = 10, 
             discr_epochs::Int64 = 5, 
             η::Float32 = 1f-3,
             fit_batch_size::Int64=64,
             ϵ::Float32 = 1f-1,
             verbose::Bool=true,
             γ::Float32=Float32(0.99),
             λ::Float32=Float32(0.97),
             KL_targ::Union{Float32,Nothing}=5f-2,
             clip_grad_tresh::Union{Float32} = 1f1,
             vf_ratio::Float32 = 5f-1,
             ent_ratio::Float32 = 0f0,
             norm_adv::Bool = true,
             use_log_rewards::Bool = false,
             stop_score::Union{Float32,Nothing}=nothing,
             trajectory_kwargs...)

    𝒟 = isnothing(initial_policy) ? ActorDiscriminator(env;γ=γ,l=l) : ActorDiscriminator(env,initial_policy;γ=γ,l=l)

    discriminator_opt = Flux.setup(Flux.Optimiser(ClipValue(clip_grad_tresh), ADAM(η)), 𝒟)
    Flux.freeze!(discriminator_opt.𝒫)
    

    policy_opt = Flux.setup(Flux.Optimiser(ClipValue(clip_grad_tresh), ADAM(η)), 𝒟)
    Flux.freeze!(policy_opt.g)
    Flux.freeze!(policy_opt.h)
    
    score_history = Vector{Float32}()
    e_losses = Vector{Float32}()
    s_losses = Vector{Float32}()
    total_d_losses = Vector{Float32}()
    for iter in 1:iterations
        
        acts_list = Vector{Vector{Vector{Float32}}}()
        states_list = Vector{Vector{Vector{Float32}}}()
        new_states_list = Vector{Vector{Vector{Float32}}}()
        scores = Vector{Float32}()
        while(length(scores)<trajectory_batch_size || length(reduce(vcat,acts_list)) < n_steps)
            env_rewards,acts,states,new_states = sas_sample(𝒟.𝒫,env;trajectory_kwargs...)
            if(use_log_rewards)
                env_rewards = dB.(env_rewards)
            end
            push!(scores,sum(env_rewards))
            
            states_list=push!(states_list,states)
            acts_list=push!(acts_list,acts)
            new_states_list=push!(new_states_list,new_states)
        
        end

        data = Flux.DataLoader((reduce(vcat,new_states_list), reduce(vcat,states_list),reduce(vcat,acts_list)), batchsize=fit_batch_size, shuffle=true, partial=false)

        d_losses = Vector{Float32}()
        for epoch in 1:discr_epochs
            
        for (batch_new_states,batch_states,batch_acts) in data
            d_loss = 0f0
            e_loss = 0f0
            s_loss = 0f0
           val, grads = Flux.withgradient(𝒟) do 𝒟
                 s_loss,e_loss = discriminator_loss(𝒟,
                            expert_states,
                            expert_acts,
                            expert_new_states,
                            batch_states,
                            batch_acts,
                            batch_new_states
                           )
                    d_loss=s_loss+e_loss

                 return -d_loss
              end

            push!(d_losses,d_loss)
            push!(s_losses,s_loss)
            push!(e_losses,e_loss)
            push!(total_d_losses,d_loss)
            Flux.update!(discriminator_opt, 𝒟, grads[1])   
            end
        end
        
        expert_D_list = [𝒟(expert_states[i],expert_acts[i],expert_new_states[i]) for i in 1:length(expert_states)]
        D_list = [𝒟(states_list[i],acts_list[i],new_states_list[i]) for i in 1:length(states_list)]
        
        rewards = [log.(x) for x in D_list] - [log.(1 .-x) for x in D_list]

        GAE_data = [GAE(states_list[i],rewards[i],acts_list[i],𝒟.𝒫;γ=γ,λ=λ) for i in 1:length(states_list)]
        rewards_to_go,Adv_list = reduce(vcat,getindex.(GAE_data,1)),reduce(vcat,getindex.(GAE_data,2))

        acts_list = reduce(vcat,acts_list)
        acts_list = Matrix{Float32}([acts_list[j][i] for i=1:size(acts_list[1])[1], j=1:size(acts_list)[1]])

        states_list = reduce(vcat,states_list)

        states_list = Matrix{Float32}([states_list[j][i] for i=1:size(states_list[1])[1], j=1:size(states_list)[1]])

        new_states_list = reduce(vcat,new_states_list)
        new_states_list = Matrix{Float32}([new_states_list[j][i] for i=1:size(new_states_list[1])[1], j=1:size(new_states_list)[1]])
        
        old_log_probs = reshape([policy_log_prob(𝒟.𝒫,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
        rewards_to_go =  Vector{Float32}(rewards_to_go)

        Adv_list =  Vector{Float32}(Adv_list)

        data = Flux.DataLoader((rewards_to_go,Adv_list,old_log_probs, states_list,acts_list), batchsize=fit_batch_size, shuffle=true, partial=false)
        
        policy_losses = Vector{Float32}()
        value_losses = Vector{Float32}()
        clip_ratios = Vector{Float32}()
        entropy_losses = Vector{Float32}()
        KL_list = Vector{Float32}()
        for epoch in 1:policy_epochs
            for (batch_rewards_to_go,batch_Adv,batch_old_log_probs,batch_states,batch_acts) in data

                
              p_loss,v_loss,entropy,clip_rat,new_log_probs,ratio =0f0,0f0,0f0,0f0,0f0,0f0
              val, grads = Flux.withgradient(𝒟) do 𝒟
                  p_loss,v_loss,entropy,clip_rat = actor_critic_Loss(
                                    𝒟.𝒫,
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

            Flux.update!(policy_opt, 𝒟, grads[1])
            
            new_log_probs = reshape([policy_log_prob(𝒟.𝒫,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
            log_ratio = new_log_probs.-old_log_probs
            KL = mean((exp.(log_ratio).- 1).-log_ratio)
            push!(KL_list,KL)

            if(!isnothing(KL_targ) && KL>1.5 * KL_targ)
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
            @printf "Mean ADV: %.5f\n" mean(Adv_list)
            @printf "Mean Clip Ratio: %.5f\n" mean(clip_ratios)
            @printf "Mean Discriminator Score: %.5f\n" mean(d_losses)
            @printf "Start Discriminator Score: %.5f\n" d_losses[1]
            @printf "End Discriminator Score: %.5f\n" d_losses[end]
            @printf "Discriminator Updates: %i\n" length(d_losses)
            @printf "Discriminator Sampled: %.5f\n" mean([mean(x) for x in D_list])
            @printf "Discriminator Expert: %.5f\n" mean([mean(x) for x in expert_D_list])
            println("-------------------------")
            flush(stdout)
        end
        push!(score_history,mean(scores))

        if(!isnothing(stop_score) && mean(scores) >= stop_score)
            break
        end
    end
    return 𝒟.𝒫,score_history,e_losses,s_losses,total_d_losses
end

function behavior_clone(𝒫::ActorCriticPolicy,expert_states::Vector{Vector{Float32}},expert_acts::Vector{Vector{Float32}};
        batchsize::Int64 = 32,η::Float32 = 1f-3,tol::Float32 = 1f-6,epochs::Int64 = 1000
)
opt = Flux.setup(ADAM(η), 𝒫)
Flux.freeze!(opt.std_network)
data = Flux.DataLoader((expert_states,expert_acts),batchsize=batchsize)
epoch_losses = Vector{Float32}()
for epoch in 1:epochs
    losses = Vector{Float32}()
    for (x,y) in data
        loss = 0f0
        val, grads = Flux.withgradient(𝒫) do 𝒫
         loss = mean(mean.([abs2.(x) for x in (𝒫.mean_network.(𝒫.feature_network.(x))-y)]))
        end
        push!(losses,loss)
        Flux.update!(opt, 𝒫, grads[1])
    end
    push!(epoch_losses,mean(losses))
    
    if(epoch % 100 == 0)
        @printf "Epoch %i\n" epoch
        @printf "Loss: %.7f\n" epoch_losses[end]
        flush(stdout)
    end
    if(epoch_losses[end]<tol)
        @printf "Epoch %i\n" epoch
        @printf "Loss: %.7f\n" epoch_losses[end]
        break
    end
end

end
