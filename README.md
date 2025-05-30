# VQA-Florence
 Task Descriptions
🔍 Subtask 1: Algorithm Development for Question Interpretation and Response
💡 Goal: This subtask requires participants to develop AI models capable of accurately interpreting and answering clinical questions based on gastrointestinal (GI) images from the Kvasir-VQA dataset. The dataset consists of 6,500 annotated images covering a range of conditions and medical instruments. Questions are categorized into six types: Yes/No, Single-Choice, Multiple-Choice, Color-Related, Location-Related, and Numerical Count, necessitating the processing of both visual and textual information. Model performance will be evaluated using multiple quantitative metrics.

✨ Focus: Create robust systems that combine image 🖼️ and text understanding 🗨️ to assist medical diagnostics 🏨.

💬 Example Questions:
🔢 How many polyps are in the image?
⚡ Are there any abnormalities in the image?
🏷️ What disease is visible in the image?

📂 Data
The 2025 dataset 🗃️ is an extended version of the HyperKvasir dataset 🔗 (datasets.simula.no/hyper-kvasir) and includes:

🏥 More images (from KVASIR-VQA) than previous years with detailed VQA annotations simulating realistic diagnostic scenarios 📝
🎯 Synthetically generated captions that can be used for image generation task. 🛠️
📥 Datasets
🏃 Development Dataset: Kvasir-VQA and captions.
🕑 Validation/Test Dataset: Can be accessed through submission/validation process (see below). This will give you the public score displayed on the leaderboard. You can split the training dataset for model development but we highly encourage to use the full development dataset in your final model.
🤫 Challenge Dataset: Private split and is very similar to development/ validation dataset. Will be used to evaluate the models. Won't be released publicly.
🧪 Evaluation Methodology
🏃 Subtask 1: Question Interpretation and Response
📊 Metrics: 📘 METEOR, 📖 ROUGE (1/2/L), and 🧠 BLEU.
📜 Evaluation: Based on correctness ✅ and relevance 📝 of answers using the provided questions 💬 and images 🖼️.
