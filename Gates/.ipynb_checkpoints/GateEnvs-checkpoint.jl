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
using Unicode
using IterTools

###Local L2 loss
function L2_loss(traj::NamedTrajectory{Float64},symb::Symbol,index::Int64,R::Float64;value::Union{Vector{Float64},Nothing}= nothing)
    return isnothing(value) ? R*traj.timestep^2/2 * sum(traj[symb][:,index].^2) : R*traj.timestep^2/2 * sum((traj[symb][:,index]-value).^2)  
end

function L2_loss(traj::NamedTrajectory{Float64},symb::Symbol,index::Vector{Int64},R::Float64;value::Union{Matrix{Float64},Nothing}= nothing)
    return isnothing(value) ? R*traj.timestep^2/2 * sum(traj[symb][:,index].^2) : R*traj.timestep^2/2 * sum((traj[symb][:,index]-value).^2)  
end

###Full Enviorment Definition of (Pre)Training Enviorments

function update!(traj::NamedTrajectory, name::Symbol, data::AbstractMatrix{Float64})
    # @assert name ∈ traj.names
    # @assert size(data, 1) == traj.dims[name]
    # @assert size(data, 2) == traj.T
    # TODO: test to see if updating both matrix and vec is necessary
    traj.data[traj.components[name], :] = data
    traj.datavec = vec(view(traj.data, :, :))
    return nothing
end

function update!(traj::NamedTrajectory, name::Symbol, idx::Int64,data::AbstractVector{Float64})
    current = deepcopy(traj[name])
    current[:,idx] = data
    update!(traj, name, current)
    return nothing
end
###Struct
Base.@kwdef mutable struct GatePretrainingEnv <: AbstractEnv
            system::AbstractQuantumSystem
            T::Int64
            𝒢::Gate
            N::Int64
            pretraining_trajectory::NamedTrajectory{Float64}
    
            dda_bound::Float64=1.0
            time_step::Float64=1/T
            
            traj::NamedTrajectory{Float64}
            ϕ⃗::Vector{Float64} = [range(0,2*pi,N)[i] for i in rand(DiscreteUniform(1,N),𝒢.n)]
end


###Construction from dynamics
function GatePretrainingEnv(system::AbstractQuantumSystem,T::Int64,𝒢::Gate,Δt::Float64,N::Int64,pretraining_trajectory::NamedTrajectory{Float64};dda_bound::Float64=1.0)
    n_controls = length(system.H_drives)

    components = (
        a = Matrix{Float64}(zeros(n_controls,T)),
        da = Matrix{Float64}(zeros(n_controls,T)),
        dda =  Matrix{Float64}(zeros(n_controls,T)),
        Ũ⃗ = reduce(hcat,[operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1)))) for i in 1:T])
    )
    
    traj = NamedTrajectory(components; timestep=Δt, controls=:a)
    return GatePretrainingEnv(
            system = system,
            T = T,
            𝒢 = 𝒢,
            N = N,
            pretraining_trajectory = pretraining_trajectory,
            dda_bound = dda_bound,
            traj = traj
            )
end

###Struct
Base.@kwdef mutable struct GateTrainingEnv <: AbstractEnv
            system::AbstractQuantumSystem
            T::Int
            𝒢::Gate
    
            dda_bound::Float64=1.0
            time_step::Float64=1/T
            
            traj::NamedTrajectory{Float64}
            ϕ⃗::Vector{Float64} = [range(0,2*pi,N)[i] for i in rand(DiscreteUniform(1,N),𝒢.n)]
end

###Construction from dynamics
function GateTrainingEnv(system::AbstractQuantumSystem,T::Int64,𝒢::Gate,Δt::Float64;dda_bound::Float64=1.0)
    n_controls = length(system.H_drives)

    components = (
        a = Matrix{Float64}(zeros(n_controls,T)),
        da = Matrix{Float64}(zeros(n_controls,T)),
        dda =  Matrix{Float64}(zeros(n_controls,T)),
        Ũ⃗ = reduce(hcat,[operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1)))) for i in 1:T])
    )
    
    traj = NamedTrajectory(components; timestep=Δt, controls=:a)
    
    return GateTrainingEnv(
            system = system,
            T = T,
            𝒢 = 𝒢,
            dda_bound = dda_bound,
            traj = traj
            )
end

#### (Shared) Env Atributes
RLBase.is_terminated(env::Union{GatePretrainingEnv,GateTrainingEnv}) = env.time_step >= (env.T-2)/env.T

RLBase.action_space(env::Union{GatePretrainingEnv,GateTrainingEnv}) = reduce(×,[(-1..1) for i in 1:length(env.system.H_drives)])

RLBase.state_space(env::Union{GatePretrainingEnv,GateTrainingEnv}) = reduce(×, [(-1..1) for i in 1:length(env.traj[:Ũ⃗][:,1])]) × reduce(×, [(-Inf..Inf) for i in 1:2*length(env.system.H_drives)]) × (1/env.T..1) × reduce(×,[(0..2*pi) for i in 1:env.𝒢.n])

RLBase.state(env::Union{GatePretrainingEnv,GateTrainingEnv})= reduce(vcat,[env.traj[:Ũ⃗][:,Int64(round(env.time_step*env.T))],env.traj[:da][:,Int64(round(env.time_step*env.T))],env.traj[:a][:,Int64(round(env.time_step*env.T))],[env.time_step],env.ϕ⃗])


###Reset Functions
function RLBase.act!(env::Union{GatePretrainingEnv,GateTrainingEnv}, action::Union{Vector{Float32},Vector{Float64}})
    t = Int64(round(env.time_step*env.T))
    action = Vector{Float64}(action)*env.dda_bound

    
    update!(env.traj, :dda, t,action)
    update!(env.traj, :a, t+1,env.traj[:a][:,t] + env.traj[:da][:,t]*env.traj.timestep)
    update!(env.traj, :da, t+1, env.traj[:da][:,t] + env.traj[:dda][:,t]*env.traj.timestep)
    update!(env.traj, :Ũ⃗, t+1,  unitary_rollout(env.traj[:Ũ⃗][:,t],hcat(env.traj[:a][:,t],zeros(length(action))),env.traj.timestep,env.system)[:,end])
    
    env.time_step += 1/env.T

    if(RLBase.is_terminated(env))
        da0 = env.traj[:da][:,t+1]
        a0 = env.traj[:a][:,t+1]
        
        dda0 =  (-a0-da0*2*env.traj.timestep)/env.traj.timestep^2
        dda1 =  (-da0-dda0*env.traj.timestep)/env.traj.timestep
        
        update!(env.traj, :dda,t+1,dda0)
        update!(env.traj, :a,  t+2,env.traj[:a][:,t+1] + env.traj[:da][:,t+1]*env.traj.timestep)
        update!(env.traj, :da, t+2,env.traj[:da][:,t+1] + env.traj[:dda][:,t+1]*env.traj.timestep)
        update!(env.traj, :Ũ⃗, t+2,unitary_rollout(env.traj[:Ũ⃗][:,t+1],hcat(env.traj[:a][:,t+1],zeros(length(action))),env.traj.timestep,env.system)[:,end])

        update!(env.traj, :dda,t+2,dda1)
        update!(env.traj, :a,  t+3,env.traj[:a][:,t+2] + env.traj[:da][:,t+2]*env.traj.timestep)
        update!(env.traj, :da, t+3,env.traj[:da][:,t+2] + env.traj[:dda][:,t+2]*env.traj.timestep)
        update!(env.traj, :Ũ⃗, t+3,unitary_rollout(env.traj[:Ũ⃗][:,t+2],hcat(env.traj[:a][:,t+2],zeros(length(action))),env.traj.timestep,env.system)[:,end])

    end
end


###Reset funcs
function RLBase.reset!(env::GatePretrainingEnv; ϕ⃗::Union{Vector{Float64},Nothing}=nothing)
    env.time_step=1/env.T

    n_controls = length(system.H_drives)
    
    update!(env.traj,:a,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:da,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:dda,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:Ũ⃗,reduce(hcat,[operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1)))) for i in 1:env.T]))
    
    env.ϕ⃗ = isnothing(ϕ⃗) ? [range(0,2*pi,env.N)[i] for i in rand(DiscreteUniform(1,env.N),env.𝒢.n)] : ϕ⃗
end

function RLBase.reset!(env::GateTrainingEnv; ϕ⃗::Union{Vector{Float64},Nothing}=nothing)
    env.time_step=1/env.T

    n_controls = length(system.H_drives)
    
    update!(env.traj,:a,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:da,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:dda,Matrix{Float64}(zeros(n_controls,env.T)))
    update!(env.traj,:Ũ⃗,reduce(hcat,[operator_to_iso_vec(Matrix{ComplexF64}(I(size(system.H_drives[1], 1)))) for i in 1:env.T]))
    
    env.ϕ⃗ = isnothing(ϕ⃗) ? rand(Uniform(0,2*pi),env.𝒢.n) : ϕ⃗
end

###Reward funcs
function RLBase.reward(env::GatePretrainingEnv,
                action::Union{Vector{Float32},Nothing};
                S::Float64=1e-2,
                Q::Float64=100.0,
                S_a::Float64=S,
                S_da::Float64=S,
                S_dda::Float64=S,
                S_U::Float64=S)
    idx = Vector{Int64}(env.ϕ⃗.*(env.N-1)/(2*pi).+1)
    idx = sum((idx[1:env.𝒢.n-1].-1).*[env.N^(env.𝒢.n-i) for i in 1:env.𝒢.n-1])+idx[end]
    if(! RLBase.is_terminated(env))
        t = Int64(round(env.time_step*env.T))
        action = Vector{Float64}(action)*env.dda_bound
        return- sum((action -env.pretraining_trajectory[Symbol("dda"*string(idx))][:,t]).^2)*Δt^2/2 * S_dda
               - L2_loss(env.traj,:a,t,S_a; value = env.pretraining_trajectory[Symbol("a"*string(idx))][:,t])
               - L2_loss(env.traj,:da,t,S_da; value = env.pretraining_trajectory[Symbol("da"*string(idx))][:,t])
               - L2_loss(env.traj,:Ũ⃗,t,S_U; value = env.pretraining_trajectory[Symbol(Unicode.normalize("Ũ⃗"*string(idx)))][:,t])
    else
        return - Q * unitary_infidelity(operator_to_iso_vec(env.𝒢(env.ϕ⃗)),env.traj[:Ũ⃗][:,end])
              - L2_loss(env.traj,:dda,[env.T-2,env.T-1,env.T],S_dda; value = env.pretraining_trajectory[Symbol("dda"*string(idx))][:,env.T-2:env.T])
               - L2_loss(env.traj,:a,[env.T-2,env.T-1,env.T],S_a; value = env.pretraining_trajectory[Symbol("a"*string(idx))][:,env.T-2:env.T])
            - L2_loss(env.traj,:da,[env.T-2,env.T-1,env.T],S_da; value = env.pretraining_trajectory[Symbol("da"*string(idx))][:,env.T-2:env.T])
             - L2_loss(env.traj,:Ũ⃗,[env.T-2,env.T-1,env.T],S_da; value = env.pretraining_trajectory[Symbol(Unicode.normalize("Ũ⃗"*string(idx)))][:,env.T-2:env.T])

    end
end


function RLBase.reward(env::GateTrainingEnv,
                action::Union{AbstractVector{Float32},Nothing};
                R::Float64=1e-2 * 2/env.traj.timestep^2,
                Q::Float64=100.0 * 2/env.traj.timestep^2,
                R_a::Float64=R,
                R_da::Float64=R,
                R_dda::Float64=R)

   if(! RLBase.is_terminated(env))
        t = Int64(round(env.time_step*env.T))
        action = Vector{Float64}(action)*env.dda_bound
        return (- sum((action).^2)*Δt^2/2 * R_dda
               - L2_loss(env.traj,:a,t,R_a)
               - L2_loss(env.traj,:da,t,R_da))
    else 
        return (- L2_loss(env.traj,:dda,[env.T-2,env.T-1,env.T],R_dda)
               - L2_loss(env.traj,:a,[env.T-2,env.T-1,env.T],R_a)
               - L2_loss(env.traj,:da,[env.T-2,env.T-1,env.T],R_da)
               - Q * unitary_infidelity(operator_to_iso_vec(env.𝒢(env.ϕ⃗)),env.traj[:Ũ⃗][:,end]))
    end
end

function get_policy_infid(env::Union{GatePretrainingEnv,GateTrainingEnv},𝒫::ActorCriticPolicy,thetas::Tuple{Float64})
    RLBase.reset!(env;ϕ⃗=[t for t in thetas])
    while(! RLBase.is_terminated(env))
        action = 𝒫(env;deterministic=true)
        RLBase.act!(env,action) 
    end
    return unitary_infidelity(operator_to_iso_vec(env.𝒢(env.ϕ⃗)),env.traj[:Ũ⃗][:,end])
end

function policy_sample(env::Union{GatePretrainingEnv,GateTrainingEnv},𝒫::ActorCriticPolicy,resolution::Int64)
    
    DATA = zeros([resolution for i in 1:env.𝒢.n]...)
    
    for (i,theta) in enumerate(product([range(0,2*pi,resolution) for i in 1:env.𝒢.n]...))
        coords = [(Int(ceil(i/resolution^(env.𝒢.n-j)))-1)%resolution+1 for j in range(1,env.𝒢.n)]
        DATA[coords...] = get_policy_infid(env,𝒫,theta)  
            
    end
    
return convert(Array{Float64,env.𝒢.n},DATA)
end
