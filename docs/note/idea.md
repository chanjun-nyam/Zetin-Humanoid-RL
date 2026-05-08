# Idea Note



## 2026/5/5

**latent state 이 가져야할 주요 property**

- state reconstruction (encoder)
    - ***note.*** we define, state = observation history
    - diagonal gaussian + cross entropy

- next observation reconstruction (forward model)
    - diagonal gaussian + cross entropy

- half periodicity
    - full 이 아닌 half periodicty 은 자연스럽게 gait symmetry 도 같이 학습
    - $0\leq\Delta t\leq T_\text{half}$에서 $f_\theta(z_{t}, z_{t+\Delta t})$와 sine wave를 BCE로 fit

        ($t$와 $t+\Delta t$에서 command가 같은 경우만)
    
    - 이러면 command가 0이여도 제자리 걸음을 할 테니 "target_vel 0 + stand" 인 command 를 따로 만들기
    - 일단 period 는 fixed 하고 학습시키기로
    - 학습 후 period 를 제외한 나머지를 fixed 시키고 최적 period를 조사 (grid search, value func. maximize)

- ***note.*** latent state 에 대한 직접적인 metric은 사용 X


**command에 따른 difficulty에 대한 고찰**

원래는 cmd vel.이더 높으면 해당 task의 difficulty가 더 높은 것이고 따라서 reward scale에 대한 조정 등이 필요하다고 생각했음.

그런데 사실 다음의 설계들이 다 적용된다면

- cmd vel.에 따른 periodic reward term들에 대한 weight 조절
- cmd vel.에 따른 energy/momentom penalty의 weight 조절

해당 모션을 만들기 위해 더 많은 에너지를 사용하는 것이 허용되는 셈이니까 해당 맥락에서는 더 어려운 task 는 아님.
인간의 직관으로는 더 (에너지 적으로)힘든 일이다 보니 더 어렵다고 보지만, 사실 사용할수 있는 에너지가 더 많아진 상태에서 해당 task의 운동학적 모션을 만들기 더 어려운가 생각해 보면 그렇다고 보기는 어려운듯.


**command design**

사람은 방향을 바꾸거나 속도를 바꿀 경우 바로 그 순간이 아니라 그 전에 먼저 준비를 함.
이는 무게중심을 변화시킨다던가, 다리 등의 각도를 바꾼다던가 하는 것들이 될 수 있음.
핵심은 운동의 변화 사전에 준비를 한다는 것임.

instant command (robot이 blinded 경우)는 이를 반영하지 못함 -> *look-ahead/time-horizon command*


**reward design**

trac lin, trac ang 등을 instant, periodic term 으로 분리해서, cmd vel. 에 따라 interpolate
energy/momentom penalty 도 cmd vel. 에 따라 rescale

trac lin, trac ang 계산시 local frame 말고 rollpitch=0 인 frame 사용하기
(그래야 rollpitch 회전으로 인해 xy 성분 velocity 가 손실되지 않고, additive 성질이 잘 성립함)

```
                        inst    periodic (full)
lin vel track (xy)      T       T
lin vel pen (z)         T       T
lin pos pen (z)         T       T
ang vel track (yaw)     T       T
ang vel pen (rollpit)   T       T           (xy)
ang pos pen (rollpit)   T       T           (xy)
torque * v
torque^2
contact
jnt limit
delt2 action^2
foot clearance
posture (필요할지 나중에 고려)
nonnegative bias
warm up reward
(foot air bonus, when ait percent < eps)
```


**개발 순서**

- reward (periodic 포함해서 다) 구현
- 일단은 instant, episode에 대해 fixed command
- 일단은 representation learning 구현 X
- deploy, 테스트
- representation learning 구현
- time-horizon, episode에서 vaiable command
    - 이에 맞춰서 reward, representation learning 수정
- inclined 상황을 고려해 reward 다시 설계 및 구현

