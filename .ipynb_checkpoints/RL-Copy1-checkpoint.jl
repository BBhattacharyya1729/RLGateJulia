Base.@kwdef mutable struct PretrainingGateEnv <: AbstractEnv
            system::AbstractQuantumSystem
            T::Int
            g::Gate
            Δt::Union{Float64, Vector{Float64}}
            N::Int64
            pretraining_trajectory::NamedTrajectory{Float64}
    
            dda_bound::Float64=1.0
            current_op::AbstractVector{Float64} = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1))))
            time_step::Float64=1/T
            
            a::AbstractMatrix{Float64} = reduce(hcat,[[0. for i in 1:length(system.H_drives)]])
            da::AbstractMatrix{Float64} = reduce(hcat,[[0. for i in 1:length(system.H_drives)]])
            dda::AbstractMatrix{Float64} = Matrix{Float64}(reshape([],length(system.H_drives),0))
            angle::Vector{Float64} = [range(0,2*pi,N)[i] for i in rand(DiscreteUniform(1,N),g.n)]

end

Base.@kwdef mutable struct TrainingGateEnv <: AbstractEnv
            system::AbstractQuantumSystem
            T::Int
            g::Gate
            Δt::Union{Float64, Vector{Float64}}
    
            dda_bound::Float64=1.0
            current_op::AbstractVector{Float64} = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1))))
            time_step::Float64=1/T
            
            a::AbstractMatrix{Float64} = reduce(hcat,[[0. for i in 1:length(system.H_drives)]])
            da::AbstractMatrix{Float64} = reduce(hcat,[[0. for i in 1:length(system.H_drives)]])
            dda::AbstractMatrix{Float64} =  Matrix{Float64}(reshape([],length(system.H_drives),0))
            angle::Vector{Float64} = rand(Uniform(0,2*pi),g.n)

end

RLBase.is_terminated(env::Union{PretrainingGateEnv,TrainingGateEnv}) = env.time_step >= (env.T-2)/env.T

RLBase.action_space(env::Union{PretrainingGateEnv,TrainingGateEnv}) = reduce(×,[(-1..1) for i in 1:length(env.system.H_drives)])

RLBase.state_space(env::Union{PretrainingGateEnv,TrainingGateEnv}) = reduce(×, [(-1..1) for i in 1:length(env.current_op)]) × reduce(×, [(-Inf..Inf) for i in 1:2*length(env.system.H_drives)]) × (1/env.T..1) × reduce(×,[(0..2*pi) for i in 1:env.g.n])

RLBase.state(env::Union{PretrainingGateEnv,TrainingGateEnv})= Vector{Float32}(reduce(vcat,[env.current_op,env.da[:,end],env.a[:,end],[env.time_step],env.angle]))

function RLBase.act!(env::Union{PretrainingGateEnv,TrainingGateEnv}, action::Vector{Float32})
    
    action = Vector{Float64}(action)*env.dda_bound
    env.dda = hcat(env.dda,action)
    env.a = hcat(env.a, env.a[:,end] + env.da[:,end]*env.Δt)
    env.da = hcat(env.da, env.da[:,end] + env.dda[:,end]*env.Δt)
    env.current_op = unitary_rollout(env.current_op,hcat(env.a[:,end],zeros(length(action))),env.Δt,env.system)[:,end]

    env.time_step += 1/env.T

    if(RLBase.is_terminated(env))
        da0 = env.da[:,end]
        a0 = env.a[:,end]
        
        dda0 = (-a0-da0*2*env.Δt)/env.Δt^2
        env.dda = hcat(env.dda, dda0)
        env.a = hcat(env.a, env.a[:,end] + env.da[:,end]*env.Δt)
        env.da = hcat(env.da, env.da[:,end] + env.dda[:,end]*env.Δt)

        dda1=(-da0-dda0*env.Δt)/env.Δt
        env.dda = hcat(env.dda, dda1)
        env.a = hcat(env.a, env.a[:,end] + env.da[:,end]*env.Δt)
        env.da = hcat(env.da, env.da[:,end] + env.dda[:,end]*env.Δt)

        env.dda = hcat(env.dda, [0. for i in 1:length(system.H_drives)])

        env.current_op = unitary_rollout(env.current_op,hcat(env.a[:,end-1:end],zeros(length(action))),env.Δt,env.system)[:,end]

    end
end

function RLBase.reset!(env::PretrainingGateEnv; angle::Union{Vector{Float64},Nothing}=nothing)
    env.current_op = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1))))
    env.time_step=1/env.T
    
    env.a = reduce(hcat,[[0. for i in 1:length(env.system.H_drives)]])
    env.da = reduce(hcat,[[0. for i in 1:length(env.system.H_drives)]])
    env.dda =  Matrix{Float64}(reshape([],length(env.system.H_drives),0))
    env.angle = isnothing(angle) ? [range(0,2*pi,N)[i] for i in rand(DiscreteUniform(1,N),env.g.n)] : angle
end

function RLBase.reset!(env::TrainingGateEnv; angle::Union{Vector{Float64},Nothing}=nothing)
    env.current_op = operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1))))
    env.time_step=1/env.T
    
    env.a = reduce(hcat,[[0. for i in 1:length(env.system.H_drives)]])
    env.da = reduce(hcat,[[0. for i in 1:length(env.system.H_drives)]])
    env.dda =  Matrix{Float64}(reshape([],length(env.system.H_drives),0))
    env.angle = isnothing(angle) ? rand(Uniform(0,2*pi),env.g.n) : angle
end

struct GatePolicy
    mean_network::Chain
    std_network::Chain
end
Flux.@functor GatePolicy

struct ACGatePolicy
    mean_network::Chain
    std_network::Chain
    VNN::Chain
end
Flux.@functor ACGatePolicy

function GatePolicy(env::Union{PretrainingGateEnv,TrainingGateEnv};l::Vector{Int64}=[16,16])
    out = length(env.system.H_drives)
    mean_in = length(RLBase.state(env))
    #std_in = env.g.n

    mean_network = Chain(Dense(mean_in=>l[1],relu),[Dense(l[i]=>l[i+1],relu) for i in 1:length(l)-1]...,Dense(l[end]=>out,softsign))
    std_network = Chain(Dense(mean_in=>l[1],relu),[Dense(l[i]=>l[i+1],relu) for i in 1:length(l)-1]...,Dense(l[end]=>1,softsign))

    return GatePolicy(mean_network,std_network)
end

function(Policy::GatePolicy)(env::Union{PretrainingGateEnv,TrainingGateEnv}; deterministic::Bool = false)
    state = Vector{Float32}(RLBase.state(env))
    means = Policy.mean_network(state)
    if(!deterministic)
        std = exp(Policy.std_network(state)[1])
        return means+rand(Normal(0,std),length(means))
    else
        return means
    end        
end

# function(Policy::GatePolicy)(state::Vector{Float32}; deterministic::Bool = false)
#     means = Policy.mean_network(state)
#     if(!deterministic)
#         std = exp(Policy.std_network(state[end-env.g.n+1:end])[1])
#         return means+rand(Normal(0,std),length(means))
#     else
#         return means
#     end        
# end

function deepcopy(Policy::GatePolicy)
    return GatePolicy(Flux.deepcopy(Policy.mean_network),Flux.deepcopy(Policy.std_network))
end

function policy_prob(policy::Union{GatePolicy,ACGatePolicy},state::Vector{Float32},action::Vector{Float32})
    means = policy.mean_network(state)
    std = exp(policy.std_network(state)[1])
    devs = action-means
    return Float32(reduce(*,exp.(-devs.^2/(2*std^2))*1/sqrt(2 * pi * std^2)))
end

function policy_log_prob(policy::Union{GatePolicy,ACGatePolicy},state::Vector{Float32},action::Vector{Float32})
    means = policy.mean_network(state)
    std = exp(policy.std_network(state)[1])
    devs = action-means
    return Float32(sum((-devs.^2/(2*std^2)).-1/2 * log(2 * pi * std^2)))
end


function RLBase.reward(env::PretrainingGateEnv;
                action::Union{AbstractVector{Float32},Nothing}=nothing,
                S::Float64=1e-2,
                S_a::Float64=S,
                S_da::Float64=S,
                S_dda::Float64=S)
    idx = Vector{Int64}(env.angle.*(env.N-1)/(2*pi).+1)
    idx = sum((idx[1:env.g.n-1].-1).*[env.N^(env.g.n-i) for i in 1:env.g.n-1])+idx[end]
    if(! RLBase.is_terminated(env))
        t = Int64(round(env.time_step*env.T))
        action = Vector{Float64}(action)*env.dda_bound
        return -(sum((env.a[:,end] - env.pretraining_trajectory[Symbol("a"*string(idx))][:,t]).^2)*Δt^2/2 * S_a+sum((env.da[:,end] -env.pretraining_trajectory[Symbol("da"*string(idx))][:,t]).^2)*Δt^2/2 * S_da + sum((action -env.pretraining_trajectory[Symbol("dda"*string(idx))][:,t]).^2)*Δt^2/2 * S_dda)
    else
        return -(sum((env.a[:,end-2:end] - env.pretraining_trajectory[Symbol("a"*string(idx))][:,end-2:end]).^2)*Δt^2/2 * S_a
            +sum((env.da[:,end-2:end] -env.pretraining_trajectory[Symbol("da"*string(idx))][:,end-2:end]).^2)*Δt^2/2 * S_da 
            + sum((env.dda[:,end-2:end] -env.pretraining_trajectory[Symbol("dda"*string(idx))][:,end-2:end]).^2)*Δt^2/2 * S_dda)
    end
end


function RLBase.reward(env::TrainingGateEnv;
                action::Union{AbstractVector{Float32},Nothing}=nothing,
                R::Float64=1e-2,
                Q::Float64=100.0,
                R_a::Float64=R,
                R_da::Float64=R,
                R_dda::Float64=R)

   if(! RLBase.is_terminated(env))
        t = Int64(round(env.time_step*env.T))
        action = Vector{Float64}(action)*env.dda_bound
        return -(sum(env.a[:,end].^2)*Δt^2/2 * R_a+sum(env.da[:,end].^2)*Δt^2/2 * R_da + sum(action.^2)*Δt^2/2 * R_dda)
    else 
        return -(sum(env.a[:,end-2:end].^2)*Δt^2/2 * R_a+sum(env.da[:,end-2:end].^2)*Δt^2/2 * R_da + sum(env.dda[:,end-2:end].^2)*Δt^2/2 * R_dda+Q*(1-abs(tr(iso_vec_to_operator(env.current_op)'env.g(env.angle))/size(env.system.H_drives[1])[1])))
    end
end

function getTrajectoryLoss(env::PretrainingGateEnv;
                S::Float64=1e-2,
                S_a::Float64=S,
                S_da::Float64=S,
                S_dda::Float64=S)
    idx = Vector{Int64}(env.angle.*(env.N-1)/(2*pi).+1)
    idx = sum((idx[1:env.g.n-1].-1).*[env.N^(env.g.n-i) for i in 1:env.g.n-1])+idx[end]
    return -(sum((env.a - env.pretraining_trajectory[Symbol("a"*string(idx))]).^2)*Δt^2/2 * S_a+sum((env.da -env.pretraining_trajectory[Symbol("da"*string(idx))]).^2)*Δt^2/2 * S_da + sum((env.dda -env.pretraining_trajectory[Symbol("dda"*string(idx))]).^2)*Δt^2/2 * S_dda)
end

function getTrajectoryLoss(env::TrainingGateEnv;
                R::Float64=1e-2,
                Q::Float64=100.0,
                R_a::Float64=R,
                R_da::Float64=R,
                R_dda::Float64=R)
    
    reg = sum(env.a.^2)*Δt^2/2 * R_a+sum(env.da.^2)*Δt^2/2 * R_da + sum(env.dda .^2)*Δt^2/2 * R_dda
    return -(reg+Q*(1-abs(tr(iso_vec_to_operator(env.current_op)'env.g(env.angle))/size(env.system.H_drives[1])[1])))
end

function SampleTrajectory(Policy::Union{GatePolicy,ACGatePolicy},env::Union{PretrainingGateEnv,TrainingGateEnv};deterministic::Bool=false, kwargs...)
    RLBase.reset!(env)
    rewards = Vector{Float64}()
    actions = Vector{Vector{Float32}}()
    states  = Vector{Vector{Float64}}()
    while(! RLBase.is_terminated(env))
        push!(states,RLBase.state(env))
        action = Policy(env;deterministic=deterministic)
        push!(actions,action)
        push!(rewards,RLBase.reward(env;action=action,kwargs...))
        RLBase.act!(env,action) 
    end
    #rewards[end]+= RLBase.reward(env;kwargs...)
    return Vector{Float32}(rewards),Vector{Vector{Float32}}(actions),Vector{Vector{Float32}}(states)
end

function euler(dda::Matrix{Float64},n_steps::Int64,Δt::Float64)
    n_controls = size(dda)[1]
    da_init=-sum(hcat([0 for i in 1:n_controls],cumsum(dda[:,1:end-1]*Δt,dims=2))[:,1:end-1],dims=2)/(n_steps-1)
    da=hcat([0 for i in 1:n_controls],cumsum(dda[:,1:end-1]*Δt,dims=2)) + reduce(hcat,[da_init for i in 1:n_steps])
    a_=hcat([0 for i in 1:n_controls],cumsum(da[:,1:end-1]*Δt,dims=2))
    return a_
end

function ValueNetwork(env::Union{PretrainingGateEnv,TrainingGateEnv};l::Vector{Int64}=[16,16])
    input = length(RLBase.state(env))
    return  Chain(Dense(input=>l[1],relu),[Dense(l[i]=>l[i+1],relu) for i in 1:length(l)-1]...,Dense(l[end]=>1))
end

discount_cumsum(l::Vector{Float32},γ::Float32) = [sum(l[j:end].*[γ^i for i in 0:length(l)-j]) for j in 1:length(l)]

function GAE(states::Vector{Vector{Float32}},rewards::Vector{Float32},actions::Vector{Vector{Float32}},VNN::Chain;γ::Float32=Float32(0.99),λ::Float32=Float32(0.97))
    vals = getindex.(VNN.(states),1)
    push!(vals,vals[end])
    δ = rewards[1:end] + γ * vals[2:end] - vals[1:end-1]
    return discount_cumsum(rewards,γ),discount_cumsum(δ,γ*λ)
end



function ACGatePolicy(Policy::GatePolicy, VNN::Chain)
    return ACGatePolicy(
           Flux.deepcopy(Policy.mean_network),
           Flux.deepcopy(Policy.std_network),
           Flux.deepcopy(VNN),
           )
end

function ACGatePolicy(env::Union{PretrainingGateEnv,TrainingGateEnv};l::Vector{Int64}=[16,16])
    return ACGatePolicy(GatePolicy(env;l),ValueNetwork(env;l))
end

function deepcopy(ACPolicy::ACGatePolicy)
    return ACGatePolicy(
           Flux.deepcopy(ACPolicy.mean_network),
           Flux.deepcopy(ACPolicy.std_network),
           Flux.deepcopy(ACPolicy.VNN),
           )
end

Flux.@functor ACGatePolicy
g(ϵ,A) = (A>=0) ? (1+ϵ) * A : (1-ϵ) * A

function ACLoss(
                ACPolicy::ACGatePolicy,
                rewards_to_go::Vector{Float32},
                states_list::Matrix{Float32},
                acts_list::Matrix{Float32},
                Adv_list::Vector{Float32},
                old_log_probs::Vector{Float32};
                ϵ::Float32 = 2f-1
               )
    value_loss = mean(abs2.(ACPolicy.VNN(states_list)'-rewards_to_go))
    new_log_probs = [policy_log_prob(ACPolicy,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)]
    policy_loss =  -mean(minimum([exp.(new_log_probs.-old_log_probs).*Adv_list g.(ϵ,Adv_list)],dims=2))
    return  policy_loss,value_loss
end
# function FitValueVNN(VNN::Chain,states_list::Matrix{Float32},rewards_to_go::Matrix{Float32}; max_iter::Int64 = 1000, lr::Float32 = 1f-3,tol::Float32=1f-4, batchsize::Int64=32)
#     epoch_losses = Vector{Float32}()
#     loss(x, y) = mean(abs2.(x.- y))
#     opt_state = Flux.setup(Adam(lr), VNN)
#     data = Flux.DataLoader((states_list, rewards_to_go), batchsize=batchsize)
    
#     for epoch in 1:max_iter
#         batch_losses = Vector{Float32}()
#         for (x_d,y_d) in data
#             val, grads = Flux.withgradient(VNN) do VNN
#               result = VNN(x_d)
#               loss(result, y_d)
#             end
        
#             # Save the loss from the forward pass. (Done outside of gradient.)
#             push!(batch_losses, val)
#             Flux.update!(opt_state, VNN, grads[1])    
#         end
#         push!(epoch_losses, mean(batch_losses))
#         if(length(epoch_losses)>2 && abs(epoch_losses[end]-epoch_losses[end-1]) <= tol)
#             break
#         end
#     end
#     return epoch_losses
# end 


# function clip_optimize(policy::GatePolicy,
#                       rewards_to_go::Matrix{Float32},
#                       states_list::Matrix{Float32},
#                       acts_list::Matrix{Float32},
#                       Adv_list::Matrix{Float32};
#                       ϵ::Float32 = 1f-1,
#                       max_iter::Int64 = 1000, lr::Float32 = 1f-3,tol::Float32=1f-4, 
#         targ_KL::Float32 = 1f-2,batchsize::Int64=32)
#     n = size(acts_list)[1]
#     old_policy = deepcopy(policy)
#     epoch_losses = Vector{Float32}()
#     opt_state = Flux.setup(Adam(lr), policy)
#     old_log_probs = reshape([policy_log_prob(old_policy,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
#     data = Flux.DataLoader((reduce(vcat,[rewards_to_go,Adv_list,old_log_probs]), vcat(states_list,acts_list)), batchsize=batchsize)
#     KL = 0
#     for epoch in 1:max_iter
#         batch_losses = Vector{Float32}()
#         for (x,y) in data
#             batch_rewards_to_go=x[1,:]
#             batch_Adv=x[2,:]
#             batch_old_log_probs=x[3,:]

#             batch_states = y[1:end-n,:]
#             batch_acts = y[end-n+1:end,:]

#             val, grads = Flux.withgradient(policy) do policy
#                 -mean(minimum([exp.([policy_log_prob(policy,batch_states[:,i],batch_acts[:,i]) for i in 1:length(batch_rewards_to_go)].-batch_old_log_probs).*batch_Adv g.(ϵ,batch_Adv)],dims=2))
#             end
#             push!(batch_losses, val)
#             Flux.update!(opt_state, policy, grads[1])   
#         end
#         push!(epoch_losses, mean(batch_losses))
#         if(length(epoch_losses)>2 && abs(epoch_losses[end]-epoch_losses[end-1]) <= tol)
#             break
#         end   
#         new_log_probs = reshape([policy_log_prob(policy,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
#         log_ratio = new_log_probs.-old_log_probs
#         KL = mean((exp.(log_ratio).- 1).-log_ratio)
#         if((KL) >= 15f-1 * targ_KL)
#             break
#         end
#     end
#     return epoch_losses,KL
# end


function PPO(env::Union{PretrainingGateEnv,TrainingGateEnv};
             trajectory_batch_size::Int64=20, 
             iterations::Int64=100,
             initial_policy::Union{ACGatePolicy,Nothing}=nothing,
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
             trajectory_kwargs...)

    ACpolicy = isnothing(initial_policy) ? ACGatePolicy(env;l=l) : initial_policy
    opt_state = Flux.setup(OptimiserChain(ClipValue(clip_grad_tresh), Adam(η)), ACpolicy)
    n = length(env.system.H_drives)

    for iter in 1:iterations
        rewards_to_go = Vector{Vector{Float32}}()
        Adv_list =   Vector{Vector{Float32}}()
        acts_list = Vector{Vector{Float32}}()
        states_list = Vector{Vector{Float32}}()
        for i in 1:trajectory_batch_size
            rewards,acts,states = SampleTrajectory(policy,env;trajectory_kwargs...)
            rtg,adv = GAE(states,rewards,acts,ACpolicy.VNN;γ=γ,λ=λ)
            
            rewards_to_go = vcat(rewards_to_go,rtg)
            Adv_list = vcat(Adv_list,adv)
            
            states_list=vcat(states_list,states)
            acts_list=vcat(acts_list,acts)
        end
        
        acts_list = Matrix{Float32}([acts_list[j][i] for i=1:size(acts_list[1])[1], j=1:size(acts_list)[1]])
        states_list = Matrix{Float32}([states_list[j][i] for i=1:size(states_list[1])[1], j=1:size(states_list)[1]])
        rewards_to_go =  Matrix{Float32}(reshape(rewards_to_go,1,length(rewards_to_go)))
        Adv_list =  Matrix{Float32}(reshape(Adv_list,1,length(Adv_list)))
        Adv_list = (Adv_list.-mean(Adv_list))/std(Adv_list)
        old_log_probs = reshape([policy_log_prob(ACpolicy,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
        data = Flux.DataLoader((reduce(vcat,[rewards_to_go,Adv_list,old_log_probs]), vcat(states_list,acts_list)), batchsize=fit_batch_size)

        policy_losses = Vector{Float32}()
        value_losses = Vector{Float32}()
        KL_list = Vector{Float32}()
        for epoch in 1:epochs
            for (x,y) in data
              batch_rewards_to_go=x[1,:]
              batch_Adv=x[2,:]
              batch_old_log_probs=x[3,:]
              batch_states = y[1:end-n,:]
              batch_acts = y[end-n+1:end,:]
              p_loss,v_loss = 0,0
              val, grads = Flux.withgradient(ACpolicy) do ACpolicy
                  p_loss,v_loss = ACLoss(
                                    ACpolicy,
                                    batch_rewards_to_go,
                                    batch_states,
                                    batch_acts,
                                    batch_Adv,
                                    batch_old_log_probs;
                                    ϵ=ϵ
                                   )
                 return vf_ratio * v_loss + p_loss
              end
            push!(policy_losses,p_loss)
            push!(value_losses,v_loss)
             Flux.update!(opt_state, ACpolicy, grads[1])
            end
            new_log_probs = reshape([policy_log_prob(ACpolicy,states_list[:,i],acts_list[:,i]) for i in 1:length(rewards_to_go)],size(Adv_list)...)
            log_ratio = new_log_probs.-old_log_probs
            KL = mean((exp.(log_ratio).- 1).-log_ratio)
            push!(KL_list,KL)

            if(!isnothing(KL_targ) && KL>KL_targ)
                break
            end
        end   
        
    
        if(verbose)
            @printf "Iterations %i Complete\n" iter
            @printf "Epochs %i \n" length(KL_list)
            @printf "Mean Rtg: %.5f\n" mean(rewards_to_go)
            @printf "Final KL: %.5f\n" KL_list[end]
            @printf "Mean Policy Loss: %.5f\n" mean(policy_losses)
            @printf "Mean Value Loss: %.5f\n" mean(value_losses)
            println("-------------------------")
            flush(stdout)
        end
    end
    return ACpolicy
end