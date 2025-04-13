import os
import tempfile
from typing import List, Dict, Any
import streamlit as st
import chromadb
import ollama
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import slide


class AIBlogCreator:
    def __init__(self, model_name="llama3", collection_name="blog_content"):
        self.model_name = model_name
        self.collection_name = collection_name

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient("./chroma_db")

        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract content from PDF file"""
        doc = fitz.open(pdf_path)
        text = ""

        # Extract text from each page
        for page in doc:
            text += page.get_text()

        return text

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF content to Markdown format"""
        text = self.extract_text_from_pdf(pdf_path)

        # Basic conversion to markdown (this is simplified)
        # A more sophisticated approach would parse headings, lists, etc.
        md_text = text.replace("\n\n", "\n\n## ")
        md_text = f"# Document: {os.path.basename(pdf_path)}\n\n{md_text}"

        return md_text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)

    def ingest_document(self, uploaded_file) -> bool:
        """Process uploaded PDF file and store in vector database"""
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(uploaded_file.getbuffer())
            pdf_path = tmp_pdf.name

        try:
            # Convert to markdown
            md_content = self.pdf_to_markdown(pdf_path)

            # Chunk the content
            chunks = self.chunk_text(md_content)

            # Store in vector db with metadata
            metadata = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

            # Add to vector store
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadata
            )

            return True
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False
        finally:
            # Clean up temp file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector database for relevant content"""
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        return results

    def generate_blog_post(self, topic: str, word_count: int = 800) -> str:
        """Generate a blog post on a specific topic"""
        # First, query the knowledge base
        relevant_docs = self.query_knowledge_base(topic)

        # Create context from retrieved documents
        context = "\n\n".join([doc[0].page_content for doc in relevant_docs])

        # Generate blog with Ollama
        prompt_system = (
            "You are an expert blog writer capable of creating engaging, informative, and well-structured blog posts. "
            "Your task is to write a blog post on the provided topic. "
            "The blog should be organized with a clear introduction, body, and conclusion. "
            "Incorporate relevant information and maintain a conversational yet professional tone."
        )

        prompt = f"Write a blog post about '{topic}' with approximately {word_count} words."
        prompt_system += f"Context: {context}"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt}
            ]
        )

        return response['message']['content']


    def generate_presentation_content(self, content: str) -> str:

        # Generate blog with Ollama
        system_prompt = (
            "You are an expert at summarizing content for presentations. "
            "Your task is to review the blog content and create a slide presentation summary. "
            "Break the content into 5-7 slide sections. Each slide should have:"
            "- A slide title (in bold with format: **Slide N: Title**)"
            "- 3 bullet points summarizing the key information (with format: *Bullet point text)"
            "The bullet points should be concise but informative."
        )

        prompt = f"Create a presentation summary for the following blog content:\n\n{content}"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return response['message']['content']


def main():
    st.set_page_config(page_title="AI Blog Content Creator", layout="wide")
    st.title("AI Blog Content Creator")
    st.write("Upload PDFs, retrieve information, and generate AI-powered blog posts")

    # Initialize the AI Blog Creator
    if 'blog_creator' not in st.session_state:
        default_model = "llama3.2:3b"
        import time
        collection_name = f"blog_content_{int(time.time())}"
        st.session_state.blog_creator = AIBlogCreator(
            model_name=default_model,
            collection_name=collection_name
        )

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "✍️ Generate Blog", "⚙️ Manage Database"])

    # Tab 1: Document Upload
    with tab1:
        st.header("Upload PDF Documents")

        # Model selection for embedding
        selected_model = st.selectbox(
            "Select Ollama Model",
            ["gemma3:latest", "qwen2.5:3b", "llama3.2:3b", "mistral:7b"],
            index=2  # Default to llama3.2:3b
        )

        if st.button("Update Model"):
            # Create a new collection with a timestamp to avoid dimension conflicts
            import time
            collection_name = f"blog_content_{int(time.time())}"
            st.session_state.blog_creator = AIBlogCreator(
                model_name=selected_model,
                collection_name=collection_name
            )
            st.success(f"Model updated to {selected_model}")

        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        uploaded_file = st.file_uploader("Limit 200MB per file • PDF", type="pdf")

        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the 5MB limit. Current size: {file_size / (1024 * 1024):.2f}MB")
            else:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        success = st.session_state.blog_creator.ingest_document(uploaded_file)
                        if success:
                            st.success(f"Document '{uploaded_file.name}' processed successfully!")

    # Tab 2: Blog Generation
    with tab2:
        st.header("Generate Blog Post")

        col1, col2 = st.columns(2)

        with col1:
            blog_topic = st.text_input("Blog Topic", placeholder="Enter the topic for your blog post")
            word_count = st.slider("Word Count", min_value=300, max_value=2000, value=800, step=100)

        with col2:
            generate_button = st.button("Generate Blog", type="primary", disabled=not blog_topic)

        if generate_button and blog_topic:
            with st.spinner(f"Generating blog about '{blog_topic}'..."):
                blog_content = st.session_state.blog_creator.generate_blog_post(
                    topic=blog_topic,
                    word_count=word_count
                )

                # Display the blog content
                st.markdown("## Generated Blog")
                st.markdown(blog_content)

                # Create download buttons in columns
                col1, col2 = st.columns(2)

                # Markdown download button
                with col1:
                    st.download_button(
                        label="Download as Markdown",
                        data=blog_content,
                        file_name=f"{blog_topic.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )

                # PDF download button
                with col2:
                    try:
                        from fpdf import FPDF
                        import tempfile

                        # Create a PDF file
                        def create_pdf_from_markdown(markdown_content, title):
                            pdf = FPDF()
                            pdf.add_page()

                            # Set font
                            pdf.set_font("Arial", size=12)

                            # Add title
                            pdf.set_font("Arial", style="B", size=16)
                            pdf.cell(200, 10, txt=title, ln=True, align='C')
                            pdf.ln(10)

                            # Reset font for body text
                            pdf.set_font("Arial", size=12)

                            # Simple markdown to PDF conversion
                            # Split content by lines
                            lines = markdown_content.split("\n")
                            for line in lines:
                                # Handle headings (crude implementation)
                                if line.startswith("# "):
                                    pdf.set_font("Arial", style="B", size=14)
                                    pdf.cell(0, 10, txt=line[2:], ln=True)
                                    pdf.set_font("Arial", size=12)
                                elif line.startswith("## "):
                                    pdf.set_font("Arial", style="B", size=12)
                                    pdf.cell(0, 10, txt=line[3:], ln=True)
                                    pdf.set_font("Arial", size=12)
                                elif line.strip() == "":
                                    pdf.ln(5)
                                else:
                                    # Handle normal text with multi-cell for wrapping
                                    pdf.multi_cell(0, 5, txt=line)

                            # Save PDF to a bytes buffer
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            return pdf_bytes

                        # Create PDF content
                        pdf_bytes = create_pdf_from_markdown(blog_content, blog_topic)

                        # Create download button for PDF
                        st.download_button(
                            label="Download as PDF",
                            data=pdf_bytes,
                            file_name=f"{blog_topic.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )

                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
                        st.info("Please install FPDF with 'pip install fpdf' to enable PDF export")

                # In the tab2 section of the app_ollama.py, modify the presentation slide section:

                # Create a presentation PDF
                st.markdown("### Presentation Slides")
                st.write("Generate a PDF presentation that summarizes the blog content:")

                # Create and offer the presentation for download
                try:
                    # Show a spinner while generating the slide content
                    with st.spinner("Generating presentation content..."):
                        slide_content = st.session_state.blog_creator.generate_presentation_content(blog_content)

                    # Show the slide content in an expander
                    with st.expander("Preview Slide Content", expanded=False):
                        st.markdown(slide_content)
                        st.markdown("---")
                        st.markdown("*This content will be formatted into slides in the PDF*")

                    # Show a spinner while creating the PDF
                    with st.spinner("Creating PDF presentation..."):
                        presentation_bytes = slide.create_presentation_pdf(blog_topic, slide_content)

                    # Success message
                    st.success("Presentation created successfully!")

                    # Add download button for the presentation
                    st.download_button(
                        label="Download Presentation PDF",
                        data=presentation_bytes,
                        file_name=f"{blog_topic.replace(' ', '_')}_presentation.pdf",
                        mime="application/pdf"
                    )

                    # Add a preview note
                    with st.expander("About this presentation", expanded=True):
                        st.info("The presentation includes:\n"
                                "- A title slide\n"
                                "- 5-7 content slides with key points from each section\n"
                                "- A closing slide")

                        # Show slide example
                        st.markdown("#### Example Slide Format")
                        st.markdown("""
                        Each slide contains:
                        - A clear section title
                        - 2-3 bullet points summarizing key information
                        - Clean, professional formatting
                        """)

                except Exception as e:
                    st.error(f"Error creating presentation: {str(e)}")
                    st.info("Please install ReportLab with 'pip install reportlab' to enable presentation generation")

    # Tab 3: Database Management
    with tab3:
        st.header("Database Management")

        # Display current database info
        st.subheader("Current Database Information")

        # Add code to get collection information compatible with ChromaDB v0.6.0+
        try:
            # Get all collections from ChromaDB
            collection_names = st.session_state.blog_creator.chroma_client.list_collections()

            st.write(f"Active Collection: {st.session_state.blog_creator.collection_name}")
            st.write(f"Current Model: {st.session_state.blog_creator.model_name}")

            # Display all collections
            if collection_names:
                st.write(f"All Collections: {', '.join(collection_names)}")

                # Get current collection count
                current_collection = st.session_state.blog_creator.collection
                item_count = current_collection.count()
                st.write(f"Documents in current collection: {item_count}")
            else:
                st.write("No collections found.")

        except Exception as e:
            st.error(f"Error retrieving database information: {str(e)}")
            st.info(
                "If using ChromaDB v0.6.0+, check the migration guide: https://docs.trychroma.com/deployment/migration")

        # Database management options
        st.subheader("Database Actions")

        col1, col2 = st.columns(2)

        with col1:
            # Option to delete current collection
            if st.button("Erase Current Collection", type="secondary"):
                try:
                    collection_name = st.session_state.blog_creator.collection_name
                    st.session_state.blog_creator.chroma_client.delete_collection(name=collection_name)

                    # Create a new collection and update the blog_creator
                    import time
                    new_collection_name = f"blog_content_{int(time.time())}"
                    st.session_state.blog_creator = AIBlogCreator(
                        model_name=st.session_state.blog_creator.model_name,
                        collection_name=new_collection_name
                    )

                    st.success(f"Collection '{collection_name}' deleted and replaced with '{new_collection_name}'")
                except Exception as e:
                    st.error(f"Error deleting collection: {str(e)}")

        with col2:
            # Dangerous option to delete all collections
            if st.button("Erase ALL Collections", type="secondary",
                         help="This will permanently delete all collections!"):
                confirm = st.checkbox("I understand this will delete all data")

                if confirm:
                    try:
                        collection_names = st.session_state.blog_creator.chroma_client.list_collections()
                        deleted_count = 0

                        for collection_name in collection_names:
                            try:
                                # Add logging to debug
                                st.write(f"Attempting to delete collection: {collection_name}")
                                st.session_state.blog_creator.chroma_client.delete_collection(name=collection_name)
                                deleted_count += 1
                            except Exception as e:
                                st.error(f"Failed to delete collection {collection_name}: {str(e)}")

                        # Create a new collection
                        import time
                        new_collection_name = f"blog_content_{int(time.time())}"
                        st.session_state.blog_creator = AIBlogCreator(
                            model_name=st.session_state.blog_creator.model_name,
                            collection_name=new_collection_name
                        )

                        # Force refresh the collection list
                        st.session_state.blog_creator.chroma_client.reset()

                        st.success(
                            f"Deleted {deleted_count} collections. Created new collection '{new_collection_name}'")

                        # Add a rerun to refresh the UI after deletion
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting all collections: {str(e)}")

if __name__ == "__main__":
    main()