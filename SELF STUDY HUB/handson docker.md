📦 محاضرة Docker - من كود إلى Container

---

🎯 الفرق بين الـ Virtual Machine والـ Container

المشكلة الأساسية

قبل ظهور الـ Virtualization والـ Containerization، كان تشغيل أكثر من تطبيق على سيرفر واحد يسبب مشاكل في التوافق بين المكتبات والعمليات، وأي تحديث قد يؤدي إلى تعطل الخدمات الأخرى . هذا أدى إلى هدر في الموارد، حيث كان لكل خدمة سيرفر منفصل.

---

الـ Virtual Machine (VM)

تعتمد على Hypervisor (مثل VMware، Hyper-V، KVM) يقوم بتقسيم الجهاز الفعلي إلى عدة أجهزة افتراضية .

الميزة التفاصيل
نظام التشغيل لكل VM نظام تشغيل كامل (Guest OS) خاص بها
العزل عزل كامل على مستوى الأجهزة (Hardware-level isolation)
حجم الملف كبير (يحتوي على نظام تشغيل كامل + التطبيقات)
وقت الإقلاع بطيء (دقائق) يحتاج إلى تشغيل نظام التشغيل بالكامل
الموارد تحجز موارد محددة حتى لو لم تستخدمها بالكامل
الأمان إذا تعرضت VM للاختراق، لا تتأثر الـ VMs الأخرى 

---

الـ Container

تعتمد على Container Engine (مثل Docker) الذي يستخدم Kernel نظام التشغيل المضيف مباشرة .

الميزة التفاصيل
نظام التشغيل لا يحتوي على Kernel خاص، بل يستخدم Kernel المضيف
العزل عزل على مستوى نظام التشغيل (OS-level isolation)
حجم الملف صغير (يحتوي فقط على التطبيق والمكتبات اللازمة)
وقت الإقلاع سريع جداً (ثوانٍ أو أجزاء من الثانية) 
الموارد موارد مشتركة ومرنة، يمكن تحديد حدود (Limits)
الأمان الخطر الأكبر هو "Container Escape" (اختراق الـ Kernel قد يؤثر على جميع الـ Containers) 

---

ملخص الفروقات 

المعيار الـ VM الـ Container
نظام التشغيل الضيف لكل VM نظام تشغيل مستقل تشترك في نفس Kernel المضيف
أمان العزل يعتمد على تنفيذ الـ Hypervisor يعتمد على Namespaces و cgroups
الأداء Overhead أعلى بسبب الترجمة Overhead قريب جداً من الصفر
وقت الإقلاع دقائق ثوانٍ أو أجزاء من الثانية
حجم التخزين كبير صغير جداً

---

🐳 ما هو Docker؟

Docker هو منصة (Platform) وليس مجرد أداة لعمل Containers. يحتوي على :

· Docker Engine: Runtime لتشغيل وإدارة الـ Containers
· Docker Images: ملفات ثابتة تحتوي على التطبيق وكل ما يحتاجه
· Docker Hub: Registry لتخزين ومشاركة الـ Images
· Docker Compose: لتنسيق عدة Containers معاً

تثبيت Docker على Linux (Ubuntu)

```
# 1. تحديث الحزم
sudo apt update

# 2. تثبيت المتطلبات
sudo apt install apt-transport-https ca-certificates curl software-properties-common

# 3. إضافة مفتاح Docker الرسمي
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 4. إضافة المستودع
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. تثبيت Docker Engine
sudo apt update
sudo apt install docker-ce

# 6. التحقق من التثبيت
docker --version
```

ملاحظة: معظم بيئات الإنتاج تعتمد على Linux، لذلك يفضل استخدام Docker Engine (بدون واجهة) بدلاً من Docker Desktop.

---

🏗️ كيف يعمل Container؟

المكونات الأساسية

· Kernel: مشترك بين جميع الـ Containers ونظام التشغيل المضيف 
· Namespace: لعزل العمليات، الشبكة، الملفات، والمستخدمين 
· Cgroups (Control Groups): للتحكم في حدود الموارد (CPU، RAM) لكل Container 

طبقات الـ Image

كل أمر في Dockerfile يضيف طبقة جديدة إلى الـ Image. ترتيب الأوامر مهم جداً لتحسين سرعة البناء:

1. الأوامر التي تتغير نادراً (مثل تثبيت الحزم الأساسية) → توضع في الأعلى
2. الأوامر التي تتغير أحياناً (مثل تثبيت المكتبات) → توضع في المنتصف
3. الأوامر التي تتغير باستمرار (مثل نسخ الكود) → توضع في الأسفل

هذا الترتيب يسمح لـ Docker بإعادة استخدام الـ Cache من الطبقات السابقة عند إعادة البناء، مما يوفر وقتاً كبيراً .

---

📝 Dockerfile عملي لنموذج ML مع FastAPI

```dockerfile
# === Stage 1: Build dependencies ===
FROM python:3.11-slim AS builder

# منع Python من كتابة ملفات .pyc ومنع التخزين المؤقت للـ stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تثبيت متطلبات النظام لتجميع بعض المكتبات
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# نسخ ملف المتطلبات أولاً للاستفادة من الـ Cache
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2: Runtime image ===
FROM python:3.11-slim AS runtime

# تثبيت مكتبات التشغيل فقط (بدون المترجمات)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مستخدم غير root للأمان
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
RUN chown appuser:appuser /app

# نسخ المكتبات المثبتة من مرحلة البناء
COPY --from=builder /install /usr/local

# نسخ الكود المصدري
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# التبديل إلى المستخدم غير root
USER appuser

# فتح المنفذ
EXPOSE 8000

# Health check للـ Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# تشغيل خادم FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

[مصدر كود مشابه مع شرح كامل لكل جزء] 

---

⚠️ نصائح مهمة لـ MLOps مع Docker

1. الفصل بين متطلبات التدريب والتشغيل

· متطلبات التدريب: MLflow، pandas، polars، DVC، Optuna، matplotlib 
· متطلبات التشغيل: FastAPI، uvicorn، pydantic (أخف وزناً ولا تحتوي على أدوات التصحيح)

2. توافق CUDA

· يجب أن تكون إصدارة CUDA داخل الـ Container متوافقة مع إصدارة Driver على الـ Host
· القاعدة: Driver المضيف ≥ CUDA في الـ Container 
· تحقق من: nvidia-smi لمعرفة إصدارة الـ Driver

3. Multi-stage Builds

تستخدم لتقليل حجم الـ Image النهائي من 15 جيجابايت إلى أقل من 2 جيجابايت :

· Stage 1 (Builder): تثبيت المترجمات والمكتبات الثقيلة
· Stage 2 (Runtime): نسخ الملفات النهائية فقط بدون المترجمات

4. .dockerignore

لمنع نسخ الملفات غير الضرورية:

```
__pycache__
*.pyc
.ipynb_checkpoints
*.log
.env
.git
data/
```

---

🔑 الخلاصة

· الـ Container أخف وأسرع من الـ VM لأنه يشارك Kernel المضيف ولا يحتوي على نظام تشغيل كامل 
· Docker هو منصة متكاملة لإدارة الـ Containers
· ترتيب الطبقات في Dockerfile يؤثر بشكل كبير على سرعة إعادة البناء
· استخدام Multi-stage builds ضروري لتقليل حجم الصور في بيئات الإنتاج 
· فصل ملفات المتطلبات بين التدريب والتشغيل يقلل الحجم ويسرع النشر 
