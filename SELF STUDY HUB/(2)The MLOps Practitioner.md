ملاحظات بداية السيشن الأولى - MLOps Course

---

تنظيم السيشن

· موعد البريك: 7:45 - 8:00 (صلاح المغرب)
· السيشن مسجلة على يوتيوب، ستُزال بعد يوم أو يومين
· الأسئلة تُكتب بعد انتهاء الشرح (وليس أثناءه) لتجنب التشتيت
· فورم الحضور موجود في وصف الفيديو، يُملأ بعد كل سيشن
· سيتم توزيع كلمة مفتاحية (Keyword) أثناء الشرح لقياس الحضور الفعلي

---

المتابعة والدعم

· ميتنقز (Google Meet) مع المجموعات بعد كل سيشن أو كل سيشنين
· تقسيم المشاركين (حتى 1200 مسجل) على فريق المودريتورز
· الميتنق هدفه: سماع الأسئلة، حل المشاكل، ترشيح موارد إضافية

---

محتوى الكورس

· الهدف: شرح Operations / Automation Cycle حول الموديلز
· لا يتم شرح Deep Learning أو LLM من الصفر
· المستهدف: Beginners و Mid-level، مع أجزاء متقدمة للSeniors
· المدة: 5 سيشنز (12-15 ساعة)، كلها عملية (Hands-on)
· جميع الأدوات المستخدمة Open Source

---

عن المحاضر

· إيه ناصر
· 7 سنوات في AI
· بدأت Computer Vision Engineer
· حالياً: Senior ML Ops Engineer في Unifonic (السعودية)
· تعمل ماسترز في جامعة النيل
· Founder: MLOps MENA Community

---

لماذا تتعلم MLOps؟

1. مطلوب في سوق العمل 2026
2. دخل أعلى من المجالات الأخرى (أعلى بــ 28% من السوق)
3. 87% من الموديلز لا تصل للإنتاج
4. كثير من المهندسين يطبقون MLOps بشكل عشوائي (موديل اسمه "final"، "final_final"، تخزين على Google Drive/WhatsApp)

---

الفرق بين:

· DevOps: ديبلوي سوفت وير (Backend, Frontend, Mobile) بشكل Trusted
· DataOps: وصول البيانات من المصدر للمستهلك بسرعة وجودة عالية
· MLOps: دمج الاثنين + التعامل مع الموديل والإنتاج + Scaling + Automation

---

3 أسئلة أساسية لأي فريق MLOps

1. هل أقدر أرجع لأي Experiment قديم من Artifact Store؟
2. الموديل بياخد قد إيه من التدريب للإنتاج؟ وهل في Pipeline Automated؟
3. مين يراقب الـ Performance و الـ Drift في Production؟

---

مراحل نضج MLOps (5 Levels)

· Level 0: شغل يدوي بالكامل (Notebooks)
· Level 1: Manual Training + Manual Deployment (بدون Tracking)
· Level 2: Automated Training Pipeline + CI + Experiment Tracking
· Level 3: Continuous Training (Auto-trigger عند وصول بيانات جديدة) + Continuous Deployment
· Level 4 (Optimal): Zero-touch Pipeline (من البداية للنهاية بدون تدخل بشري)

---

سؤال Interview مهم

السؤال: عندك فريق مكون من 4 ML Developers، كل واحد شغال بـ Framework مختلف (TensorFlow, PyTorch, Scikit-learn, Keras)، الداتا على لاب توب كل واحد، ومافيش Standardization، والتسليم يتم عن طريق واتساب أو Google Drive. المشاكل إيه والحلول إيه؟

المشاكل المذكورة:

· Static Paths / Hardcoded Paths
· بيانات غير Consistent
· لكل مطور Framework مختلف
· مافيش Version Control للبيانات أو الموديلز
· مافيش تكامل أو تسليم منظم

الحلول المبدئية:

· استخدام Environment Variables أو Config Files بدل الـ Hardcoding
· استخدام أداة إدارة Secrets زي Vault
· توحيد الـ Framework أو استخدام Standardized Wrapper
· استخدام Artifact Store و Model Registry
· استخدام Git + DVC للبيانات والموديلز

---

كلمة سر الجلسة (Keyword)

لم تذكر بعد، ستُقال أثناء الشرح لاحقاً
