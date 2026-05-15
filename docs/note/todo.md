# **To-do Lists**


## **Implementation**
**twist-swing decomposition**

- angular velocity decompose 하는법
- `articulation_data_manager`, `observation_manager`, `reward_manager` 반영
- twist linear velocity 는 incline을 고려하지 않는데, 이를 어떻게 할지

**RL Environment**

known problem
학습 도중 sampling std sample 할때 std 가 negative or nan 임
- env 문제 x, observation/action manager 의 tensor debugger 에서 검출 안됨
- 그러면 ppo, 또는 일종의 steep gradient(?) 문제 같은데 (td3 에서 뭐 비슷한 내용 있었던것 같은데) -> obs noise 추가 하기


module 마다 non check, log 관리

nan 발생 시 env capture + model capture + nan 발생 action acpture

torch.jit 고려

observation manager > n_obs != env.n_obs <<<<<<<<<<<<<<<<< 검토

obs/reward clip

TensorDebugger 문서화

- reward
  - periodical reward
    - last period command 의 consistency 를 같이 고려

- terrain curriculum
  - GTC 고려
  - terrain type 에 따라 command sampling 다르게
  - terrain type 에 따라 disturbance 다르게

- command
  - period

- randomize

- termination
```
<Uniform Termination>

Uniform Episode Offset

step_index

offset: 1, 2, 3, 4, 5 
->
offset: 1, 2, 3, 2, 5


offset: 1, 1, 1, 1, 1
offset: 1, 0, 1, 2, 1
```

**RL algorithm**

- representation learning
  - ppo warmup stages (encoder, forward model 등의 초기 학습을 위해)

**scripts**

run scripts
- functionalities
  - train, play, test, benchmark (terrain mesh 조각내기)
- Typer 이용해 구현

vscode workspace setup script

- make vscode .settings "python.analysis.extraPaths"
- make other workspace path dependent scripts (ex. load_isaacsim_package.sh)

**Logging**

- logtree -> log & model 관련 새 패키지로 빼기
  - w&b, tensorboard 통합
  - hyperparameter 도 포함
  - path manager / log data unit


## **Documentation & Management**

- README.MD
- SETUP.MD
  - Visual Studio Code workspace
- ADR.MD
- python docstring documentation
- version 관리
- license
- contributing
- reward manager docstring
