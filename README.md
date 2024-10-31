147.46.121.38
ID : aidas_intern_1
PW : 1qaz2wsx!!

서버 접속 후 다음 순서대로 입력
(1) unset DISPLAY
(2) aidas_a6gpu -g=1

우리 연구실의 A6000 사용 가능. 그러나 공용으로 사용하기 때문에 다른 사람이 사용하고 있으면 대기해야함.

그럴 때 (2)는 아래의 명령어들로도 대체 가능(다른 A6000 or A100 GPU이지만 이건 다른 연구실과도 공유하고 있는 GPU라서 할당이 더 오래 걸릴 가능성이 있음)
- shr_a6gpu -g=1 
- shr1_agpu -g=1 
- mig_agpu -g=1

qstat -u $USER
qdel pid