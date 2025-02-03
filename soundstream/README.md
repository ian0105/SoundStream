# SoundStream


특이 사항
- pytorch-lightning을 활용해 구현하였습니다.
- 아래 방식을 통해 필요 라이브러리를 설치할 수 있습니다.
```
poetry shell
poetry update
```
- 학습은 아래 코드로 실행할 수 있습니다.
```
python main fit --config/base.yaml
```
- Hyperparameters는 config/base.yaml에서 수정할 수 있습니다.
- Inference 예시는 아래 코드로 실행할 수 있습니다.
```
python infer.py "checkpoint 경로" "inference 데이터"
```

