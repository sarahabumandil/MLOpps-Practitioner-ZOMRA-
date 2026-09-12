"""The bilingual knowledge base — the whole cost of the Arabic lesson.

Forty notes on this session's own material, each written in English and Arabic.
That is deliberately the entire infrastructure for the bilingual chapter: no
vector database, no second embedding model, no external corpus. Those belong to
the final project, not to a 25-minute lab.

Why bilingual at all. Slide 53 claims an LLM judge is weaker at Arabic claim
decomposition than at English. `evals/calibrate_judge.py` reproduces that on the
student's own laptop, as a measured Cohen's kappa gap rather than a bullet
point. Nobody teaching this session in English can show that.

Two things about Arabic text that are load-bearing here:

* **Tokenization.** ollama_langfuse_rag.py's retriever matched `[a-z0-9]+`,
  which finds nothing at all in Arabic script. The tokenizer below is
  Unicode-aware, so Arabic retrieval works before it is broken on purpose.
* **Normalisation.** Arabic writes the same word several ways — alef with or
  without hamza, teh marbuta versus heh, optional diacritics, decorative
  tatweel. Search systems normalise these away. The rule is that the index and
  the query must be normalised IDENTICALLY; incident 09 breaks exactly that
  invariant and only Arabic notices.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: One note, two languages. Keys are language codes so `docs_for(lang)` is a
#: dict lookup rather than a branch.
Note = dict[str, Any]

NOTES: list[Note] = [
    {
        "id": "psi",
        "en": (
            "Population Stability Index",
            "PSI measures how far a distribution has moved from a baseline. Below 0.10 the "
            "population is stable. Between 0.10 and 0.25 is a moderate shift worth "
            "investigating. Above 0.25 is significant and is the conventional trigger to "
            "retrain. Unlike a p-value, PSI is an effect size, so it does not inflate with "
            "sample size.",
        ),
        "ar": (
            "مؤشر استقرار التوزيع",
            "يقيس مؤشر PSI مقدار ابتعاد التوزيع الحالي عن التوزيع المرجعي. أقل من 0.10 يعني "
            "أن التوزيع مستقر. بين 0.10 و0.25 يعني تغيرًا متوسطًا يستحق الفحص. أكثر من 0.25 "
            "يعني تغيرًا كبيرًا وهو الحد المتعارف عليه لإعادة التدريب. وعلى عكس قيمة p، فإن "
            "PSI مقياس لحجم الأثر ولا يتضخم بزيادة حجم العينة.",
        ),
    },
    {
        "id": "ks",
        "en": (
            "Kolmogorov-Smirnov test",
            "The KS test compares two cumulative distributions and reports the largest gap "
            "between them as a p-value. Use it for continuous features such as trip "
            "distance. Because it is a significance test, a large enough sample makes even "
            "a meaningless shift come out significant.",
        ),
        "ar": (
            "اختبار كولموجوروف-سميرنوف",
            "يقارن اختبار KS بين توزيعين تراكميين ويُبلغ عن أكبر فجوة بينهما على شكل قيمة p. "
            "يُستخدم مع الخصائص المستمرة مثل مسافة الرحلة. ولأنه اختبار دلالة إحصائية، فإن "
            "العينة الكبيرة تجعل حتى التغير التافه يبدو ذا دلالة.",
        ),
    },
    {
        "id": "chisquare",
        "en": (
            "Chi-square test",
            "The chi-square test asks whether observed category counts differ from expected "
            "counts by more than chance allows. Use it for low-cardinality categorical "
            "features such as passenger count.",
        ),
        "ar": (
            "اختبار مربع كاي",
            "يسأل اختبار مربع كاي عمّا إذا كانت أعداد الفئات المرصودة تختلف عن الأعداد "
            "المتوقعة بأكثر مما تسمح به الصدفة. يُستخدم مع الخصائص الفئوية قليلة التنوع مثل "
            "عدد الركاب.",
        ),
    },
    {
        "id": "concept-drift",
        "en": (
            "Concept drift",
            "Concept drift is a change in the relationship between inputs and the target, "
            "and is invisible to input or output drift checks. It only shows up in the "
            "error, so it needs ground truth labels. Page-Hinkley catches sustained "
            "directional shifts; ADWIN shrinks an adaptive window on abrupt change.",
        ),
        "ar": (
            "انحراف المفهوم",
            "انحراف المفهوم هو تغير في العلاقة بين المدخلات والهدف، ولا تستطيع فحوص انحراف "
            "المدخلات أو المخرجات رؤيته. لا يظهر إلا في الخطأ، ولذلك يحتاج إلى تسميات حقيقية. "
            "يكتشف Page-Hinkley التحولات المستمرة في اتجاه واحد، بينما يقلّص ADWIN نافذته "
            "المتكيفة عند التغير المفاجئ.",
        ),
    },
    {
        "id": "drift-order",
        "en": (
            "Order of detection",
            "Feature drift is observable immediately and needs no labels. Prediction drift "
            "comes next and also needs no labels. Error drift is the ground truth that the "
            "model is broken, but it arrives last because labels lag predictions by hours "
            "or weeks.",
        ),
        "ar": (
            "ترتيب الاكتشاف",
            "انحراف الخصائص يمكن رصده فورًا ولا يحتاج إلى تسميات. ثم يأتي انحراف التنبؤات "
            "وهو أيضًا لا يحتاج إلى تسميات. أما انحراف الخطأ فهو الدليل القاطع على أن النموذج "
            "معطل، لكنه يصل أخيرًا لأن التسميات تتأخر عن التنبؤات ساعات أو أسابيع.",
        ),
    },
    {
        "id": "prometheus-types",
        "en": (
            "Prometheus metric types",
            "A histogram accumulates observations into buckets and only goes up; use it for "
            "latency and predicted duration. A gauge is a single value that moves both "
            "ways; use it for a PSI score recomputed each run. Never average a latency.",
        ),
        "ar": (
            "أنواع مقاييس Prometheus",
            "الهيستوجرام يجمّع الملاحظات في صناديق ولا يتناقص أبدًا، ويُستخدم لزمن الاستجابة "
            "والمدة المتوقعة. أما المقياس اللحظي (gauge) فقيمة واحدة تتحرك صعودًا وهبوطًا، "
            "ويُستخدم لدرجة PSI المعاد حسابها كل تشغيل. لا تحسب متوسط زمن الاستجابة أبدًا.",
        ),
    },
    {
        "id": "histogram-quantile",
        "en": (
            "Percentiles from histograms",
            "Compute a percentile with histogram_quantile over the _bucket series, summing "
            "by le before applying the quantile. Averaging pre-computed percentiles across "
            "instances is arithmetically meaningless.",
        ),
        "ar": (
            "حساب المئينات من الهيستوجرام",
            "احسب المئين باستخدام histogram_quantile على سلسلة _bucket، مع الجمع حسب le قبل "
            "تطبيق المئين. أما حساب متوسط المئينات المحسوبة مسبقًا عبر عدة نسخ فهو بلا معنى "
            "حسابيًا.",
        ),
    },
    {
        "id": "counter-rate",
        "en": (
            "Why counters need rate()",
            "A counter only increases and resets to zero when the process restarts. Its raw "
            "value answers nothing; rate() converts it to a per-second change and handles "
            "the reset correctly.",
        ),
        "ar": (
            "لماذا يحتاج العداد إلى rate",
            "العداد يزداد فقط ويعود إلى الصفر عند إعادة تشغيل العملية. قيمته الخام لا تجيب "
            "عن شيء، بينما تحوّلها دالة rate إلى معدل تغير في الثانية وتعالج إعادة الضبط "
            "بشكل صحيح.",
        ),
    },
    {
        "id": "cardinality",
        "en": (
            "Metric cardinality",
            "Every distinct combination of label values is a separate time series. Putting "
            "an unbounded identifier such as a ride id in a label mints one series per "
            "request and will exhaust the server's memory.",
        ),
        "ar": (
            "تعدد قيم الوسوم في المقاييس",
            "كل توليفة مختلفة من قيم الوسوم تُنشئ سلسلة زمنية منفصلة. وضع معرّف غير محدود "
            "مثل معرّف الرحلة في وسم يُنشئ سلسلة لكل طلب وسيستنزف ذاكرة الخادم.",
        ),
    },
    {
        "id": "sample-limit",
        "en": (
            "sample_limit",
            "sample_limit rejects a scrape carrying more samples than the limit, whole, and "
            "marks the target down. That turns a server-wide out-of-memory failure into one "
            "loud, local, recoverable one that names itself in the targets page.",
        ),
        "ar": (
            "حد العينات sample_limit",
            "يرفض sample_limit عملية السحب بالكامل إذا تجاوزت عدد العينات المسموح، ويضع "
            "الهدف في حالة توقف. وهذا يحوّل انهيار الذاكرة على مستوى الخادم كله إلى عطل واحد "
            "محلي وواضح وقابل للتعافي، يعلن عن نفسه في صفحة الأهداف.",
        ),
    },
    {
        "id": "frozen-reference",
        "en": (
            "The frozen reference",
            "Drift must be measured against a fixed baseline, normally the training set. If "
            "you compare today against a rolling recent window, the baseline chases the "
            "drift and a slow shift silently becomes the new normal.",
        ),
        "ar": (
            "المرجع المجمّد",
            "يجب قياس الانحراف مقابل خط أساس ثابت، وعادةً ما يكون مجموعة التدريب. أما إذا "
            "قارنت اليوم بنافذة حديثة متحركة، فإن خط الأساس يلاحق الانحراف ويتحول التغير "
            "البطيء بصمت إلى وضع طبيعي جديد.",
        ),
    },
    {
        "id": "effect-size",
        "en": (
            "Effect size versus p-value",
            "A p-value answers whether a difference could be chance; at production traffic "
            "volumes the answer is always no. An effect size answers how big the difference "
            "is, and does not inflate with sample size. Alert on effect sizes.",
        ),
        "ar": (
            "حجم الأثر مقابل قيمة p",
            "تجيب قيمة p عمّا إذا كان الفرق قد يكون مصادفة، وعند أحجام المرور الإنتاجية تكون "
            "الإجابة دائمًا لا. أما حجم الأثر فيجيب عن مقدار الفرق ولا يتضخم بزيادة حجم "
            "العينة. اجعل تنبيهاتك على أحجام الأثر.",
        ),
    },
    {
        "id": "segments",
        "en": (
            "Segment monitoring",
            "A global average is volume-weighted, so a segment at eight percent of traffic "
            "cannot move it. Split every quality metric by city, model version, language "
            "and prompt version. A healthy average is the most common way a real failure "
            "stays invisible.",
        ),
        "ar": (
            "مراقبة الشرائح",
            "المتوسط العام مرجّح بالحجم، لذلك لا تستطيع شريحة تمثل ثمانية بالمئة من المرور "
            "تحريكه. قسّم كل مقياس جودة حسب المدينة وإصدار النموذج واللغة وإصدار التوجيه. "
            "المتوسط السليم هو الطريقة الأشيع لبقاء عطل حقيقي غير مرئي.",
        ),
    },
    {
        "id": "late-labels",
        "en": (
            "Late labels",
            "Ground truth arrives hours or weeks after the prediction. Until it does, "
            "accuracy cannot be computed at all, which is why input and output drift are "
            "monitored first.",
        ),
        "ar": (
            "التسميات المتأخرة",
            "تصل الحقيقة الأرضية بعد ساعات أو أسابيع من التنبؤ. وإلى أن تصل، لا يمكن حساب "
            "الدقة إطلاقًا، ولهذا تُراقَب انحرافات المدخلات والمخرجات أولًا.",
        ),
    },
    {
        "id": "train-serve-skew",
        "en": (
            "Train/serve skew",
            "Train/serve skew is when the serving path transforms inputs differently from "
            "the training path. No drift monitor can see it, because the inputs are "
            "unchanged. Only replaying logged serving vectors through the offline pipeline "
            "finds it.",
        ),
        "ar": (
            "التفاوت بين التدريب والتشغيل",
            "يحدث التفاوت بين التدريب والتشغيل عندما يعالج مسار التشغيل المدخلات بطريقة "
            "مختلفة عن مسار التدريب. لا يستطيع أي مراقب انحراف رؤيته لأن المدخلات لم تتغير. "
            "ولا يكشفه إلا إعادة تمرير متجهات التشغيل المسجلة عبر المسار غير المباشر.",
        ),
    },
    {
        "id": "replay",
        "en": (
            "Replay as a detector",
            "Replay takes the exact feature vectors the service scored, re-scores them "
            "offline, and compares. If the two disagree on identical inputs, the difference "
            "is the serving path itself.",
        ),
        "ar": (
            "إعادة التشغيل كأداة كشف",
            "تأخذ إعادة التشغيل متجهات الخصائص نفسها التي قيّمتها الخدمة، وتعيد تقييمها خارج "
            "الخدمة، ثم تقارن. فإذا اختلفت النتيجتان على مدخلات متطابقة، كان الفرق هو مسار "
            "التشغيل نفسه.",
        ),
    },
    {
        "id": "retrain-gate",
        "en": (
            "The retrain gate",
            "Detecting drift is not sufficient grounds to retrain. The gate also requires "
            "that the drift is sustained, the upstream pipeline is healthy, enough new "
            "labels have arrived, and the cooldown since the last retrain has passed.",
        ),
        "ar": (
            "بوابة إعادة التدريب",
            "اكتشاف الانحراف ليس سببًا كافيًا لإعادة التدريب. تشترط البوابة أيضًا أن يكون "
            "الانحراف مستمرًا، وأن يكون خط البيانات سليمًا، وأن تصل تسميات جديدة كافية، وأن "
            "تنقضي فترة التهدئة منذ آخر إعادة تدريب.",
        ),
    },
    {
        "id": "seasonality",
        "en": (
            "Seasonal drift",
            "Ramadan, Eid and quarter end produce drift that is real, sustained and "
            "properly labelled, and that reverts in weeks. A model refitted to it will be "
            "wrong for the rest of the year, so a business calendar must block automatic "
            "promotion.",
        ),
        "ar": (
            "الانحراف الموسمي",
            "تنتج فترات مثل رمضان والعيد ونهاية الربع المالي انحرافًا حقيقيًا ومستمرًا وموثقًا "
            "بالتسميات، لكنه يزول خلال أسابيع. النموذج المعاد تدريبه عليه سيكون خاطئًا بقية "
            "العام، ولذلك يجب أن يمنع التقويم التجاري الترقية التلقائية.",
        ),
    },
    {
        "id": "random-split-trap",
        "en": (
            "The random split trap",
            "Validating a challenger on a random split of an anomalous window shares the "
            "anomaly between both halves, so the challenger scores brilliantly against the "
            "very regime it overfitted. Validate on data from after the window.",
        ),
        "ar": (
            "فخ التقسيم العشوائي",
            "التحقق من نموذج منافس باستخدام تقسيم عشوائي لنافذة شاذة يجعل الشذوذ مشتركًا بين "
            "نصفيها، فيحقق النموذج نتيجة ممتازة مقابل النظام نفسه الذي بالغ في ملاءمته. تحقق "
            "من النموذج ببيانات من بعد انتهاء النافذة.",
        ),
    },
    {
        "id": "cooldown",
        "en": (
            "Retrain cooldown",
            "A cooldown stops the pipeline retraining faster than late labels can evaluate "
            "the previous model. Without it a sustained drift signal ships unvalidated "
            "models on a timer.",
        ),
        "ar": (
            "فترة التهدئة بين عمليات إعادة التدريب",
            "تمنع فترة التهدئة إعادة التدريب بوتيرة أسرع مما تستطيع التسميات المتأخرة تقييم "
            "النموذج السابق. وبدونها تدفع إشارة انحراف مستمرة نماذج غير متحقق منها إلى "
            "الإنتاج بشكل دوري.",
        ),
    },
    {
        "id": "data-quality",
        "en": (
            "Pipeline health checks",
            "Retraining on the output of a broken pipeline does not fix the model, it "
            "launders the bug into the weights where no data-quality check will look "
            "again. Schema, freshness, null rate and value ranges are checked first.",
        ),
        "ar": (
            "فحوص سلامة خط البيانات",
            "إعادة التدريب على مخرجات خط بيانات معطل لا تصلح النموذج، بل تُدخل الخلل إلى "
            "الأوزان حيث لن ينظر إليه أي فحص جودة مرة أخرى. لذلك تُفحص البنية والحداثة ونسبة "
            "القيم الفارغة ونطاقات القيم أولًا.",
        ),
    },
    {
        "id": "unit-change",
        "en": (
            "Unit changes pass validation",
            "A feed that switches from kilometres to miles still sends valid floats in a "
            "float field, so every schema validator passes it. Only a distribution check or "
            "an explicit range assertion catches a unit change.",
        ),
        "ar": (
            "تغير الوحدات يجتاز التحقق",
            "مصدر البيانات الذي يتحول من الكيلومترات إلى الأميال ما زال يرسل أرقامًا عشرية "
            "صالحة في حقل عشري، فيجتاز كل مدققات البنية. ولا يكشف تغير الوحدة إلا فحص توزيع "
            "أو تأكيد صريح على نطاق القيم.",
        ),
    },
    {
        "id": "deploy-annotation",
        "en": (
            "Deploy annotations",
            "Most incidents are changes. Marking every deploy on the dashboard lets you line "
            "a symptom's start time against what changed, which is usually the fastest path "
            "to a cause.",
        ),
        "ar": (
            "تعليقات النشر على اللوحات",
            "معظم الحوادث تغييرات. وضع علامة لكل عملية نشر على لوحة المراقبة يتيح لك مطابقة "
            "وقت بداية العَرَض مع ما تغيّر، وهو غالبًا أسرع طريق إلى السبب.",
        ),
    },
    {
        "id": "canary",
        "en": (
            "Canary releases",
            "A canary sends a small share of traffic to a new version and compares it "
            "against the current one. An unfinished canary leaves two versions live, and "
            "every aggregate metric becomes a blend of two models.",
        ),
        "ar": (
            "الإصدار التجريبي المحدود",
            "يوجّه الإصدار التجريبي المحدود نسبة صغيرة من المرور إلى نسخة جديدة ويقارنها "
            "بالنسخة الحالية. والإصدار التجريبي غير المكتمل يترك نسختين تعملان معًا، فيصبح كل "
            "مقياس إجمالي مزيجًا من نموذجين.",
        ),
    },
    {
        "id": "alert-severity",
        "en": (
            "Page on symptoms, warn on causes",
            "A symptom is something a user is experiencing now, such as the service being "
            "down. A cause is something that will hurt later, such as drift climbing. Page "
            "on causes and people learn to ignore the pager.",
        ),
        "ar": (
            "نبّه فورًا على الأعراض وحذّر على الأسباب",
            "العَرَض شيء يعانيه المستخدم الآن، مثل توقف الخدمة. أما السبب فشيء سيؤذي لاحقًا، "
            "مثل تصاعد الانحراف. إذا نبّهت فورًا على الأسباب تعلّم الناس تجاهل التنبيهات.",
        ),
    },
    {
        "id": "runbook",
        "en": (
            "Runbooks",
            "Every alert should link to instructions written before the incident, by someone "
            "who was not woken up. An alert with no runbook is a pager with no instructions "
            "attached.",
        ),
        "ar": (
            "أدلة الاستجابة",
            "يجب أن يرتبط كل تنبيه بتعليمات كُتبت قبل وقوع الحادثة، بواسطة شخص لم يُوقَظ من "
            "نومه. التنبيه بلا دليل استجابة هو استدعاء بلا تعليمات.",
        ),
    },
    {
        "id": "postmortem",
        "en": (
            "Postmortems",
            "A postmortem separates what happened from how you found out. The useful "
            "question is not what broke but why it took so long to see, and whether the "
            "signal was missing, aggregated away, or present and unread.",
        ),
        "ar": (
            "تحليل ما بعد الحادثة",
            "يفصل تحليل ما بعد الحادثة بين ما حدث وبين كيفية اكتشافه. والسؤال المفيد ليس ما "
            "الذي تعطل، بل لماذا استغرقت رؤيته كل هذا الوقت، وهل كانت الإشارة غائبة أم "
            "مطموسة في التجميع أم موجودة ولم تُقرأ.",
        ),
    },
    {
        "id": "predict-linear",
        "en": (
            "Predicting exhaustion",
            "predict_linear extrapolates a trend to say when a resource runs out. A disk at "
            "eighty-five percent tells you nothing; a disk losing four gigabytes an hour "
            "tells you when it dies. This is the only kind of alert that fires before the "
            "outage.",
        ),
        "ar": (
            "التنبؤ بنفاد الموارد",
            "تستقرئ دالة predict_linear الاتجاه لتحدد متى ينفد المورد. القرص الممتلئ بنسبة "
            "خمسة وثمانين بالمئة لا يخبرك بشيء، أما القرص الذي يفقد أربعة جيجابايت في الساعة "
            "فيخبرك متى سيتوقف. وهذا النوع الوحيد من التنبيهات الذي ينطلق قبل العطل.",
        ),
    },
    {
        "id": "dead-feedback",
        "en": (
            "Monitor the monitoring",
            "If the feedback path silently stops, no labels arrive, and error metrics freeze "
            "at their last good value. A frozen quality metric looks exactly like a healthy "
            "one, so alert on the rate of incoming labels itself.",
        ),
        "ar": (
            "راقب أدوات المراقبة نفسها",
            "إذا توقف مسار التغذية الراجعة بصمت، فلن تصل أي تسميات وتتجمد مقاييس الخطأ عند "
            "آخر قيمة سليمة. ومقياس الجودة المتجمد يبدو تمامًا مثل السليم، لذلك نبّه على معدل "
            "وصول التسميات نفسه.",
        ),
    },
    {
        "id": "three-pillars",
        "en": (
            "Metrics, logs and traces",
            "Metrics tell you that something is wrong and are cheap to keep. Logs tell you "
            "what happened in one request. Traces tell you where the time and the causality "
            "went across many steps. You need all three for different questions.",
        ),
        "ar": (
            "المقاييس والسجلات والتتبعات",
            "تخبرك المقاييس بأن شيئًا ما خاطئ وتكلفة الاحتفاظ بها منخفضة. وتخبرك السجلات بما "
            "حدث داخل طلب واحد. أما التتبعات فتخبرك أين ذهب الوقت وكيف تسلسلت الأسباب عبر "
            "خطوات متعددة. وأنت تحتاج الثلاثة لأسئلة مختلفة.",
        ),
    },
    {
        "id": "tracing",
        "en": (
            "Nested tracing",
            "An LLM trace should nest retrieval, reranking and generation as separate "
            "observations. Most reported hallucinations turn out to be retrieval failures, "
            "and you can only see that if retrieval is its own span.",
        ),
        "ar": (
            "التتبع المتداخل",
            "يجب أن يتضمن تتبع نموذج اللغة خطوات الاسترجاع وإعادة الترتيب والتوليد كملاحظات "
            "منفصلة متداخلة. ومعظم ما يُبلَّغ عنه كهلوسة يتبين أنه فشل في الاسترجاع، ولا يمكنك "
            "رؤية ذلك إلا إذا كان الاسترجاع مقطعًا مستقلًا.",
        ),
    },
    {
        "id": "scores",
        "en": (
            "Evaluation scores",
            "Scores attach a judgement to a trace and can be numeric, boolean, categorical "
            "or free text. They come from three sources: cheap programmatic checks, an LLM "
            "judge, and human review. Keep the source in the score name.",
        ),
        "ar": (
            "درجات التقييم",
            "تربط الدرجات حكمًا بالتتبع، وقد تكون رقمية أو منطقية أو فئوية أو نصًا حرًا. وهي "
            "تأتي من ثلاثة مصادر: فحوص برمجية رخيصة، وحَكَم من نماذج اللغة، ومراجعة بشرية. "
            "احتفظ بالمصدر داخل اسم الدرجة.",
        ),
    },
    {
        "id": "prompt-versioning",
        "en": (
            "Prompt versioning",
            "A prompt is a deployable artifact. Moving a production label to a new version "
            "changes behaviour with no code deploy and no restart, so the deploy timeline is "
            "empty. If prompts are not versioned they are not observable.",
        ),
        "ar": (
            "إصدارات التوجيهات",
            "التوجيه أحد مكونات النشر. ونقل وسم الإنتاج إلى إصدار جديد يغيّر السلوك دون نشر "
            "كود ودون إعادة تشغيل، فيبقى سجل النشر فارغًا. وإذا لم تكن التوجيهات مُصدَّرة "
            "فهي غير قابلة للمراقبة.",
        ),
    },
    {
        "id": "llm-judge",
        "en": (
            "LLM as a judge",
            "An LLM judge scores outputs against a written rubric. It is only useful if it "
            "is pinned: same model tag, same digest, same temperature, same seed and same "
            "context window. A score is only comparable to another score from the same "
            "ruler.",
        ),
        "ar": (
            "نموذج اللغة كحَكَم",
            "يقيّم الحَكَم من نماذج اللغة المخرجات وفق معيار مكتوب. وهو لا يفيد إلا إذا كان "
            "مثبتًا: نفس وسم النموذج ونفس البصمة ونفس درجة الحرارة ونفس البذرة ونفس حجم "
            "نافذة السياق. فالدرجة لا تقارَن إلا بدرجة أخرى من المسطرة نفسها.",
        ),
    },
    {
        "id": "judge-calibration",
        "en": (
            "Judge calibration",
            "Before trusting a judge, measure its agreement with human labels using Cohen's "
            "kappa, which corrects for agreement by chance. Below about 0.80 the judge is "
            "not measuring what you think. Report it separately per language.",
        ),
        "ar": (
            "معايرة الحَكَم",
            "قبل الوثوق بالحَكَم، قِس مدى اتفاقه مع التسميات البشرية باستخدام معامل كابا "
            "لكوهين الذي يصحح الاتفاق الناتج عن الصدفة. وتحت 0.80 تقريبًا لا يقيس الحَكَم ما "
            "تظنه. واذكر النتيجة لكل لغة على حدة.",
        ),
    },
    {
        "id": "faithfulness",
        "en": (
            "Faithfulness",
            "Faithfulness decomposes an answer into atomic claims and checks each one "
            "against the retrieved context, scoring supported claims over total claims. It "
            "measures grounding, not correctness: an answer can be perfectly faithful to a "
            "context that is itself wrong.",
        ),
        "ar": (
            "الأمانة للسياق",
            "تفكك الأمانة الإجابة إلى ادعاءات ذرية وتتحقق من كل واحد منها مقابل السياق "
            "المسترجع، فتكون الدرجة عدد الادعاءات المدعومة على إجماليها. وهي تقيس الاستناد "
            "لا الصحة: فقد تكون الإجابة أمينة تمامًا لسياق خاطئ في ذاته.",
        ),
    },
    {
        "id": "response-relevancy",
        "en": (
            "Response relevancy",
            "Response relevancy generates several questions from the answer, embeds them, "
            "and measures cosine similarity to the real question. A legitimate refusal "
            "scores low, so refusals must be bucketed separately rather than dragging the "
            "mean down.",
        ),
        "ar": (
            "ملاءمة الإجابة",
            "تولّد ملاءمة الإجابة عدة أسئلة من الإجابة نفسها، ثم تمثّلها متجهيًا وتقيس تشابه "
            "جيب التمام مع السؤال الأصلي. والرفض المشروع يحصل على درجة منخفضة، لذلك يجب عزل "
            "حالات الرفض في فئة مستقلة بدل أن تخفض المتوسط.",
        ),
    },
    {
        "id": "context-precision",
        "en": (
            "Context precision and recall",
            "Context precision asks whether the retrieved chunks were actually useful. "
            "Context recall asks whether retrieval found everything relevant, which needs a "
            "reference answer, so it cannot run on live traffic.",
        ),
        "ar": (
            "دقة السياق واستدعاؤه",
            "تسأل دقة السياق عمّا إذا كانت المقاطع المسترجعة مفيدة فعلًا. أما استدعاء السياق "
            "فيسأل عمّا إذا كان الاسترجاع قد وجد كل ما هو ذو صلة، وهو يحتاج إجابة مرجعية، "
            "ولذلك لا يمكن تشغيله على المرور الحي.",
        ),
    },
    {
        "id": "nan-rate",
        "en": (
            "NaN rate",
            "When a judge's output will not parse, the metric library retries and then drops "
            "the sample. The dropped samples are the hard ones, so the surviving mean looks "
            "better than reality. Always report the NaN rate next to the score.",
        ),
        "ar": (
            "نسبة القيم المفقودة",
            "عندما يتعذر تحليل مخرجات الحَكَم، تعيد مكتبة القياس المحاولة ثم تسقط العينة. "
            "والعينات المسقطة هي الأصعب، فيبدو المتوسط الباقي أفضل من الواقع. لذلك اذكر دائمًا "
            "نسبة القيم المفقودة بجوار الدرجة.",
        ),
    },
    {
        "id": "num-ctx",
        "en": (
            "Context window truncation",
            "Ollama defaults to a 2048-token context and truncates silently. An evaluation "
            "sends the answer plus every retrieved chunk, and Arabic uses more tokens per "
            "word than English, so Arabic hits the ceiling first and its claims are marked "
            "unsupported for reasons unrelated to the application.",
        ),
        "ar": (
            "اقتطاع نافذة السياق",
            "يستخدم Ollama نافذة سياق افتراضية بحجم 2048 رمزًا ويقتطع ما يزيد عنها بصمت. "
            "وعملية التقييم ترسل الإجابة وكل المقاطع المسترجعة، والعربية تستهلك رموزًا أكثر "
            "لكل كلمة من الإنجليزية، فتصطدم بالسقف أولًا وتُوسم ادعاءاتها بأنها غير مدعومة "
            "لأسباب لا علاقة لها بالتطبيق.",
        ),
    },
    {
        "id": "embedding-drift",
        "en": (
            "Embedding drift",
            "Track the mean cosine similarity of query embeddings over time, or the maximum "
            "mean discrepancy between two periods. If users start asking a different kind of "
            "question, the retriever may no longer match their intent.",
        ),
        "ar": (
            "انحراف التمثيلات المتجهية",
            "تابع متوسط تشابه جيب التمام لتمثيلات الاستعلامات عبر الزمن، أو التباين الأقصى "
            "للمتوسطات بين فترتين. فإذا بدأ المستخدمون يسألون نوعًا مختلفًا من الأسئلة، فقد لا "
            "يعود المسترجع مطابقًا لمقصدهم.",
        ),
    },
    {
        "id": "text-normalisation",
        "en": (
            "Text normalisation must match",
            "Search normalises text before indexing: case folding, diacritic removal, letter "
            "unification. The index and the query must be normalised identically. Normalise "
            "only one side and lookups silently miss, and in Arabic this breaks alef and teh "
            "marbuta variants first.",
        ),
        "ar": (
            "توحيد صيغة النص يجب أن يتطابق",
            "يوحّد البحث صيغة النص قبل الفهرسة: توحيد حالة الأحرف وإزالة التشكيل وتوحيد "
            "الحروف. ويجب توحيد صيغة الفهرس والاستعلام بالطريقة نفسها. فإذا وحّدت طرفًا واحدًا "
            "فقط فشلت عمليات البحث بصمت، وفي العربية تنكسر أولًا صور الألف والتاء المربوطة.",
        ),
    },
    {
        "id": "cost-tracking",
        "en": (
            "Cost tracking for self-hosted models",
            "A self-hosted model has no per-token price, so cost dashboards read zero. The "
            "honest unit is GPU-hours times instance price divided by tokens produced. "
            "Report the minimum and maximum week, never the average, because budgets are set "
            "by the peak.",
        ),
        "ar": (
            "تتبع التكلفة للنماذج المستضافة ذاتيًا",
            "النموذج المستضاف ذاتيًا ليس له سعر لكل رمز، لذلك تقرأ لوحات التكلفة صفرًا. "
            "والوحدة الصادقة هي ساعات المعالج الرسومي مضروبة في سعر الجهاز مقسومة على عدد "
            "الرموز المنتجة. واذكر الأسبوع الأدنى والأعلى، لا المتوسط أبدًا، لأن الميزانيات "
            "تُوضع على الذروة.",
        ),
    },
    {
        "id": "programmatic-scorers",
        "en": (
            "Programmatic scorers",
            "Cheap deterministic checks can run on one hundred percent of traffic: valid "
            "JSON, a non-empty retrieval, a citation present, a length limit respected. Run "
            "these everywhere and reserve the expensive judge for a sample.",
        ),
        "ar": (
            "المقيّمات البرمجية",
            "الفحوص الحتمية الرخيصة يمكن تشغيلها على مئة بالمئة من المرور: صحة صيغة JSON، "
            "ووجود نتائج استرجاع غير فارغة، ووجود استشهاد، واحترام حد الطول. شغّل هذه في كل "
            "مكان واحتفظ بالحَكَم المكلف لعينة فقط.",
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════
#  Normalisation and tokenization
# ══════════════════════════════════════════════════════════════════════

#: Harakat (fatha, damma, kasra, sukun, shadda, tanwin) plus the superscript
#: alef. Arabic is normally written without them, so a document that has them
#: and a query that does not will not match unless both sides are normalised.
ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟ]")
#: Tatweel/kashida — a decorative letter-stretching character with no meaning.
TATWEEL = "ـ"
#: Alef variants unify to bare alef; teh marbuta to heh; alef maqsura to yeh.
#: This is what every Arabic search stack does, and it is correct — as long as
#: it is done on BOTH sides of the index.
LETTER_FOLDING = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",  # أ إ آ -> ا
        "ة": "ه",  # ة -> ه
        "ى": "ي",  # ى -> ي
    }
)

#: Unicode-aware. The retriever in ollama_langfuse_rag.py matched `[a-z0-9]+`,
#: which finds precisely nothing in Arabic script — so Arabic retrieval returned
#: zero hits for every query before this existed. That is not the lesson we
#: want; incident 09 is.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its of on or "
    "should that the this to use used using what when which why with you your "
    "من في على عن الى إلى هو هي هذا هذه ذلك التي الذي ما لا أن إن كان يكون مع كل "
    "أو و ثم قد لكن حتى بين عند بعد قبل".split()
)


def normalise(text: str, fold: bool = True) -> str:
    """Normalise text for indexing or querying.

    `fold=True` applies the Arabic-specific folding: strip diacritics and
    tatweel, unify alef/teh-marbuta/alef-maqsura. On Latin script every one of
    those steps is a no-op, which is exactly why incident 09 breaks Arabic and
    leaves English untouched.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    if fold:
        text = ARABIC_DIACRITICS.sub("", text).replace(TATWEEL, "")
        text = text.translate(LETTER_FOLDING)
    return text


#: Light stemming: Arabic attaches the definite article to the word, so "التوزيع"
#: and "توزيع" are the same term. Every serious Arabic search stack strips it.
#: Only on words long enough that the remainder is still a word.
_AL_PREFIX = re.compile(r"^(ال)(\w{3,})$")


def stem(token: str) -> str:
    """Strip the Arabic definite article. A no-op on Latin script."""
    match = _AL_PREFIX.match(token)
    return match.group(2) if match else token


def tokenize(text: str, fold: bool = True) -> list[str]:
    """Normalise, split into words, drop stopwords, and light-stem when folding.

    `fold` controls BOTH the letter folding and the stemming, because they are
    the same decision: they are the normalisation the index applies. The rule
    that matters is not which steps you choose — it is that the query must apply
    exactly the same ones.
    """
    words = _WORD_RE.findall(normalise(text, fold=fold))
    if fold:
        words = [stem(w) for w in words]
    return [w for w in words if w not in _STOPWORDS]


def detect_language(text: str) -> str:
    """'ar' if the text contains Arabic letters, else 'en'.

    Crude on purpose: every score in this repo is reported per language, so what
    matters is that the split is consistent, not that it handles every script.
    """
    return "ar" if any("؀" <= ch <= "ۿ" for ch in text) else "en"


def docs_for(lang: str = "en") -> list[dict[str, str]]:
    """The corpus in one language, in the {id, title, text} shape the RAG uses."""
    out = []
    for note in NOTES:
        title, text = note[lang]
        out.append({"id": note["id"], "title": title, "text": text, "lang": lang})
    return out


def note_by_id(note_id: str, lang: str = "en") -> dict[str, str] | None:
    """One note, in one language."""
    for note in NOTES:
        if note["id"] == note_id:
            title, text = note[lang]
            return {"id": note_id, "title": title, "text": text, "lang": lang}
    return None


def index_is_asymmetric() -> bool:
    """True while incident 09 is active: the index folds and the query does not.

    Normalising one side of an index and not the other is a real, common bug and
    it fails SILENTLY — no error, no empty-index warning, just lookups that stop
    matching. In Arabic it is immediate, because folding changes almost every
    word. In English it changes nothing, so English keeps working perfectly and
    the global retrieval metric barely moves.
    """
    try:
        from incidents import state

        return bool(state.flag("KB_NORMALISE_INDEX_ONLY"))
    except ImportError:
        return False


def index_tokens(text: str) -> list[str]:
    """Tokens as stored in the index. Always folded — this side is not the bug."""
    return tokenize(text, fold=True)


def query_tokens(text: str) -> list[str]:
    """Tokens as looked up. Folded to match the index, unless incident 09 is on."""
    return tokenize(text, fold=not index_is_asymmetric())


__all__ = [
    "NOTES",
    "docs_for",
    "note_by_id",
    "detect_language",
    "normalise",
    "tokenize",
    "index_tokens",
    "query_tokens",
    "index_is_asymmetric",
]
