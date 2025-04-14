# AI Blog Creator 📝

**AI Blog Creator** is a Streamlit-based application that allows users to generate high-quality blog posts and presentation slides using Ollama and OpenAI's GPT models. The app also supports RAG by uploading PDF documents to enhance the AI's knowledge base for blog generation.
---

### Two versions of the app are available: one using OpenAI's API and another using Ollama's local model. This README provides instructions for setting up and using the OpenAI version. The changes required to switch to the Ollama version are minimal and will be noted in the relevant sections.

##  Online Demo 

You can try out the application online at [AI Blog Content Creator Demo](https://aiblogcontentcreator.streamlit.app/).

## 🚀 Features

- **Blog Generation**: Create engaging and well-structured blog posts by providing a topic and word count.
- **PDF Upload**: Upload PDF documents to enhance the AI's knowledge base for more context-aware blog generation.
- **Content Download**: Download generated blogs as Markdown or PDF files.
- **Presentation Slides**: Generate a summarized presentation in PDF format from the blog content.
- **Database Management**: Manage the vector database (ChromaDB) for storing and retrieving document embeddings.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai-blog-creator.git
   cd ai-blog-creator
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root directory.
   - Add the following line to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
5. **Install Ollama (if using the Ollama version):**
   - Follow the instructions on the [Ollama website](https://ollama.com/) to install and set up the local model.
   - Make sure to have the Ollama model downloaded and running locally.
   - Update the code in app_ollama.py to use your local model.
   - Make sure to run the app_ollama.py file instead of app_openai.py.


6. **Run the application:**
   ```bash
   streamlit run app_openai.py
   ```

---

## 📚 Usage

### Generate Blog

1. Navigate to the **"Generate Blog"** tab.
2. Enter a blog topic and word count.
3. *(Optional)* Upload PDF documents to enhance the AI's knowledge base.
4. Click **"Generate Blog"** to create the blog content.

### View and Download Content

- Switch to the **"Generated Content"** tab to view the blog.
- Download the blog as **Markdown** or **PDF**.
- Generate and download a **presentation** summarizing the blog content.

### Manage Database

- Use the **"Database Management"** tab to view, delete, or reset the vector database.

---

## 📦 Requirements

- Python 3.8 or higher  
- Streamlit  
- OpenAI Python SDK  
- ChromaDB  
- FPDF  
- PyMuPDF (`fitz`)  
- `python-dotenv`

---

## 📁 File Structure

```
ai-blog-creator/
├── app_openai.py           # Main application file
├── requirements.txt        # List of dependencies
├── .env                    # Environment variables (not included in repo)
├── slide.py                # Helper module for creating presentation PDFs
```

---

## 🧰 Troubleshooting

- **Error creating PDF**: Ensure FPDF is installed  
  ```bash
  pip install fpdf
  ```

- **Error generating presentation**:  
  ```bash
  pip install reportlab
  ```

- **Embedding dimension mismatch**: Ensure consistent embedding models are used in the code.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- [OpenAI](https://openai.com/) for their GPT models  
- [Streamlit](https://streamlit.io/) for the web application framework  
- [ChromaDB](https://www.trychroma.com/) for vector database management  
- [FPDF](https://pyfpdf.github.io/) for PDF generation