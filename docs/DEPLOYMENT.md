# 배포 가이드

## 목차

1. [Docker 배포](#docker-배포)
2. [Kubernetes 배포](#kubernetes-배포)
3. [Linux 서버 배포](#linux-서버-배포)
4. [프로덕션 체크리스트](#프로덕션-체크리스트)
5. [모니터링 설정](#모니터링-설정)
6. [자동 백업](#자동-백업)

---

## Docker 배포

### 1단계: 이미지 빌드

```bash
# 로컬 빌드
docker build -t rag-system:latest .

# 태그 지정 (선택)
docker tag rag-system:latest rag-system:1.0.0
```

### 2단계: Docker Compose로 배포

```bash
# 프로덕션 환경 설정
ENVIRONMENT=production docker-compose up -d

# 상태 확인
docker-compose ps
docker-compose logs -f ui
```

### 3단계: 스케일링 (선택)

```bash
# API 서버 여러 개 실행
docker-compose up -d --scale api=3

# 로드 밸런서 설정 필요
# nginx.conf 참고
```

---

## Kubernetes 배포

### 전제 조건

- Kubernetes 클러스터 (1.20+)
- kubectl 설치
- Helm (선택)

### 1단계: 이미지 푸시

```bash
# Docker Registry에 푸시
docker tag rag-system:latest your-registry/rag-system:1.0.0
docker push your-registry/rag-system:1.0.0
```

### 2단계: 매니페스트 적용

```bash
# 네임스페이스 생성
kubectl create namespace rag-system

# 설정 맵 생성
kubectl create configmap rag-config \
  --from-file=.env \
  -n rag-system

# 배포
kubectl apply -f k8s/deployment.yaml -n rag-system

# 상태 확인
kubectl get pods -n rag-system
kubectl logs -f deployment/rag-system -n rag-system
```

### 3단계: 서비스 노출

```bash
# 로드 밸런서 생성
kubectl apply -f k8s/service.yaml -n rag-system

# 외부 IP 확인
kubectl get service -n rag-system
```

### K8s 매니페스트 예제 (k8s/deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-system
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-system
  template:
    metadata:
      labels:
        app: rag-system
    spec:
      containers:
      - name: rag-ui
        image: your-registry/rag-system:1.0.0
        ports:
        - containerPort: 8501
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama:11434"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## Linux 서버 배포

### 전제 조건

- Ubuntu 20.04+ 또는 CentOS 8+
- Python 3.11+
- Ollama 서버

### 1단계: 환경 설정

```bash
# 사용자 생성
sudo useradd -m -s /bin/bash rag

# 디렉토리 생성
sudo mkdir -p /opt/rag-system
sudo chown -R rag:rag /opt/rag-system

# 저장소 클론
cd /opt/rag-system
sudo -u rag git clone https://github.com/darkzard05/rag-system-ollama.git .
```

### 2단계: 의존성 설치

```bash
# Python 환경
cd /opt/rag-system
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3단계: Systemd 서비스 설정

```bash
# /etc/systemd/system/rag-system.service 생성
sudo cat > /etc/systemd/system/rag-system.service << 'EOF'
[Unit]
Description=RAG System Service
After=network.target

[Service]
Type=simple
User=rag
WorkingDirectory=/opt/rag-system
Environment="PATH=/opt/rag-system/venv/bin"
ExecStart=/opt/rag-system/venv/bin/streamlit run src/ui.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable rag-system
sudo systemctl start rag-system

# 상태 확인
sudo systemctl status rag-system
```

### 4단계: Nginx 리버스 프록시

```bash
# /etc/nginx/sites-available/rag-system 생성
sudo cat > /etc/nginx/sites-available/rag-system << 'EOF'
upstream streamlit {
    server localhost:8501;
}

upstream api {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://streamlit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api/ {
        proxy_pass http://api/;
        proxy_set_header Host $host;
    }
}
EOF

# 활성화
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5단계: SSL 설정 (Let's Encrypt)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 프로덕션 체크리스트

### 보안

- [ ] `.env` 파일의 모든 키 변경
- [ ] JWT_SECRET_KEY 강력한 키로 설정
- [ ] RBAC 활성화
- [ ] 캐시 암호화 활성화
- [ ] SSL/TLS 인증서 설정
- [ ] 방화벽 규칙 설정 (필요한 포트만 개방)
- [ ] 정기적 보안 업데이트

### 성능

- [ ] 캐시 활성화 (Redis 권장)
- [ ] 배치 사이즈 최적화
- [ ] 로드 밸런서 설정
- [ ] CDN 설정 (정적 파일용)
- [ ] 데이터베이스 인덱싱

### 모니터링

- [ ] Prometheus 설정
- [ ] Grafana 대시보드 구성
- [ ] 경보 규칙 설정
- [ ] 로그 수집 (ELK/Loki)
- [ ] 에러 추적 (Sentry)

### 백업 & 복구

- [ ] 자동 백업 설정
- [ ] 복구 테스트 수행
- [ ] 재해 복구 계획 수립
- [ ] 데이터 보존 정책 정의

### 문서화

- [ ] 배포 절차 문서화
- [ ] 비상 연락처 정보 정리
- [ ] 운영 매뉴얼 작성
- [ ] 문제 해결 가이드 작성

---

## 모니터링 설정

### Prometheus 설정

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics'
```

### Grafana 대시보드

1. Grafana 접속: http://localhost:3000 (admin/admin)
2. Prometheus 데이터 소스 추가
3. 대시보드 import:
   - `./monitoring/grafana-dashboards/rag-system.json`

### 경보 설정

```yaml
# monitoring/alerts.yml
groups:
  - name: rag_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate below 50%"
```

---

## 자동 백업

### 데이터베이스 백업

```bash
# 일일 백업 스크립트 (backup.sh)
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/rag-system"
mkdir -p $BACKUP_DIR

# 벡터 저장소 백업
cp -r /app/cache/vector_store $BACKUP_DIR/vector_store_$DATE

# 문서 백업
cp -r /app/data/documents $BACKUP_DIR/documents_$DATE

# 이전 백업 삭제 (30일 이상)
find $BACKUP_DIR -mtime +30 -delete
```

### Cron 작업 설정

```bash
# 매일 자정에 백업 실행
0 0 * * * /opt/rag-system/backup.sh
```

### 복구 절차

```bash
# 특정 날짜 백업에서 복구
BACKUP_DATE="20250121_000000"
cp -r /backups/rag-system/vector_store_$BACKUP_DATE/* /app/cache/vector_store/
cp -r /backups/rag-system/documents_$BACKUP_DATE/* /app/data/documents/

# 서비스 재시작
docker-compose restart ui
```

---

## 성능 튜닝

### Docker 메모리 제한

```yaml
# docker-compose.yml
services:
  ui:
    mem_limit: 8g
    memswap_limit: 8g
```

### Ollama GPU 설정

```bash
# GPU 활용 (NVIDIA)
CUDA_VISIBLE_DEVICES=0 ollama serve

# 또는 .env에서
OLLAMA_NUM_GPU=1
```

### 캐시 최적화

```bash
# Redis 메모리 제한
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

---

## 문제 해결

### 배포 후 서비스 접근 불가

```bash
# 로그 확인
docker-compose logs ui

# 포트 확인
netstat -an | grep LISTEN

# 방화벽 확인
sudo ufw status
sudo ufw allow 8501
```

### 높은 메모리 사용

```bash
# 메모리 사용량 확인
docker stats

# 불필요한 이미지 정리
docker image prune -a

# 캐시 정리
curl -X POST http://localhost:8000/api/v1/cache/clear
```

### 느린 응답 시간

```bash
# 메트릭 확인
curl http://localhost:8000/api/v1/metrics | jq

# 프로파일링 실행
python -m cProfile src/ui.py
```

---

**마지막 업데이트:** 2025-01-21
