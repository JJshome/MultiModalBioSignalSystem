# 다중 모달 생체 자극 시스템 예제

이 디렉토리에는 다중 모달 생체 자극 진단 및 치료 시스템의 다양한 사용 예제가 포함되어 있습니다. 각 예제는 시스템의 특정 기능과 활용 시나리오를 보여줍니다.

## 다중 기기 이상 감지 예제 (multi_device_anomaly_detection.py)

이 예제는 복수의 무선 기기를 통해 수집된 생체 신호에서 이상 거동을 감지하고, 이에 대응하여 자극 파라미터를 조정하는 방법을 보여줍니다. 특허 기술의 핵심 개념인 다중 모달 통합 및 폐루프(closed-loop) 제어 시스템을 구현하였습니다.

### 기능 설명

- **다중 생체 신호 시뮬레이션**: ECG, EMG, PPG 신호를 실시간으로 생성
- **이상 패턴 주입**: 각 신호에 특정 시점에 다양한 이상 패턴을 인위적으로 주입
- **트랜스포머 기반 이상 감지**: 실시간으로 신호 패턴을 분석하여 이상 거동 감지
- **적응형 자극 파라미터 조정**: 감지된 이상에 따라 TENS 자극 파라미터 자동 조정
- **종합 보고서 생성**: 분석 결과 및 추천 사항을 포함한 보고서 생성

### 사용 방법

```bash
# 기본 사용법
python multi_device_anomaly_detection.py

# 시뮬레이션 시간 지정 (초 단위)
python multi_device_anomaly_detection.py --duration 60

# 출력 디렉토리 지정
python multi_device_anomaly_detection.py --output custom_output_dir

# 실제 기기 연결 모드 사용 (시뮬레이션 대신)
python multi_device_anomaly_detection.py --real
```

### 출력 결과

이 예제는 다음과 같은 출력을 생성합니다:

1. **콘솔 로그**: 감지된 이상 및 조정된 파라미터에 대한 실시간 로그
2. **종합 보고서**: `[output_dir]/report_overview.png`
3. **신호별 이상 감지 시각화**: `[output_dir]/report_[signal_type]_anomalies.png`
4. **통합 신호 시각화**: `[output_dir]/multi_device_signals.png`

### 맞춤 설정

예제 코드에서 다음 부분을 수정하여 테스트 시나리오를 변경할 수 있습니다:

1. **이상 패턴 설정**:
```python
self.anomaly_settings = {
    "ecg": {"enabled": True, "start": 10.0, "duration": 5.0, "amplitude": 0.5},
    "emg": {"enabled": True, "start": 15.0, "duration": 3.0, "amplitude": 0.8},
    "ppg": {"enabled": True, "start": 18.0, "duration": 2.0, "amplitude": 0.4}
}
```

2. **TENS 자극 기본 파라미터**:
```python
self.stimulation_params = {
    "frequency": 50.0,  # Hz
    "pulse_width": 200.0,  # µs
    "intensity": 20.0,  # mA
    "mode": "continuous",
    "duration": 30.0,  # seconds
    # ...
}
```

3. **신호 파라미터**:
```python
self.ecg_params = {
    "sampling_rate": 200,  # Hz
    "heart_rate": 60,  # BPM
    "heart_rate_variability": 5.0,  # BPM
    "noise_level": 0.05
}
```

### 시나리오 예시

이 예제는 다음과 같은 실제 사용 시나리오를 시뮬레이션합니다:

1. **부정맥 감지 및 대응**: ECG 신호에서 부정맥 패턴 감지 후 TENS 자극 강도 및 주파수 자동 조정
2. **근육 긴장 관리**: EMG에서 과도한 근육 활성화 감지 후 TENS 펄스 폭 조정
3. **혈류 이상 개선**: PPG에서 혈류 감소 패턴 감지 후 혈류 개선 모드로 자극 변경

## 추가 예제 개발 예정

다음과 같은 추가 예제가 개발 예정입니다:

- **수면 모니터링 및 개선**: EEG, HRV 및 호흡 패턴을 분석하여 수면 질 개선을 위한 자극 제공
- **스트레스 관리**: GSR, HRV, 호흡 패턴을 분석하여 스트레스 수준에 따른 이완 자극 제공
- **운동 성능 최적화**: 운동 중 EMG, ECG, 움직임 데이터를 분석하여 근육 퍼포먼스 최적화 자극 제공
- **인지 기능 향상**: EEG 기반 인지 상태 분석 및 최적 집중력을 위한 신경 자극 제공
