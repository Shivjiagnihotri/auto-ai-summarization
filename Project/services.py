import streamlit as st
import re 
import time
import base64
from tika import parser
import warnings
warnings.simplefilter('ignore')
import time
from io import BytesIO
from stqdm import stqdm
from transformers import pipeline
from transformers import BertTokenizer

# Getting a pdf
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas

# Streamlit app code
st.set_page_config(
    page_title='AI Summarizer',
    page_icon='🌼',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
    'About': " Developed by Shivji Agnihotri. 🦄🤓"
    }
)

hide_streamlit_style = """
<style>

    /* Hide the page expander */
    div[data-testid='stSidebarNav'] ul {max-height:none}

    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    button[title="View fullscreen"]{
    visibility: hidden;}

    div.block-container{padding-top:2rem;}

    div[class^='css-1544g2n'] { padding-top: 0rem; }
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0.605rem;
    }
    .stDeployButton, footer, #stDecoration {
        visibility: hidden;
    }

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.title("Select an option")

# from PIL import Image
# image = Image.open('logo.jpeg')
# st.sidebar.image(image, width=None , use_column_width=True, clamp=True, channels="RGB", output_format="auto") #caption= 'Creating a Sangria Experience'


# Session State also supports the attribute based syntax
if 'summary' not in st.session_state:
    st.session_state.summary = None

@st.cache_resource(show_spinner=False)
def extract_doc_text(pdf_path):
    document_text = ""
    # try:
    parsed = parser.from_file(pdf_path)
    document_text = parsed["content"]
    document_text = re.sub('([ \t]+)|([\n]+)', lambda m: ' ' if m.group(1) else '\n', document_text)
    return document_text

# @st.cache_resource(show_spinner=False)
def prep_b4_save(text):
    text = re.sub('Gods', 'God\'s', text)
    text = re.sub('yours', 'your\'s', text)
    text = re.sub('dont', 'don\'t', text)
    text = re.sub('doesnt', 'doesn\'t', text)
    text = re.sub('isnt', 'isn\'t', text)
    text = re.sub('havent', 'haven\'t', text)
    text = re.sub('hasnt', 'hasn\'t', text)
    text = re.sub('wouldnt', 'wouldn\'t', text)
    text = re.sub('theyre', 'they\'re', text)
    text = re.sub('youve', 'you\'ve', text)
    text = re.sub('arent', 'aren\'t', text)
    text = re.sub('youre', 'you\'re', text)
    text = re.sub('cant', 'can\'t', text)
    text = re.sub('whore', 'who\'re', text)
    text = re.sub('whos', 'who\'s', text)
    text = re.sub('whatre', 'what\'re', text)
    text = re.sub('whats', 'what\'s', text)
    text = re.sub('hadnt', 'hadn\'t', text)
    text = re.sub('didnt', 'didn\'t', text)
    text = re.sub('couldnt', 'couldn\'t', text)
    text = re.sub('theyll', 'they\'ll', text)
    text = re.sub('youd', 'you\'d', text)
    return text
# @st.cache_resource(show_spinner=False)
def text_chunking(new_text, size_of_chunk): #, size_of_chunk
    max_chunk = size_of_chunk#250
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    words = new_text.split(' ')
    current_chunk = 0
    chunks = []
    for word in words:
        if len(chunks) == current_chunk + 1:
            # Check the number of tokens instead of the number of words
            if len(tokenizer.encode(' '.join(chunks[current_chunk]), truncation=False)) + len(tokenizer.encode(word, truncation=False)) <= max_chunk:
                chunks[current_chunk].append(word)
            else:
                current_chunk += 1
                chunks.append([word])
        else:
            chunks.append([word])

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks
@st.cache_resource(show_spinner=False)
def transformers_summary(chunks, max_length, min_length, ): #ngram, temp, topk, beams
    # global summary_length
    global summarizer
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum") # sshleifer/distilbart-xsum-6-6 pszemraj/led-large-book-summary  pszemraj/led-base-book-summary  pszemraj/long-t5-tglobal-base-16384-book-summary philschmid/bart-large-cnn-samsum #knkarthick/MEETING_SUMMARY, facebook/bart-large-cnn, Falconsai/medical_summarization, Falconsai/text_summarization lidiya/bart-large-xsum-samsum
    
    bulletedSummaryString = "Here are the key insights:\n"

    st.toast("🚀 **Summarizing the text. Please wait...**")
    with st.spinner("Fetching key insights.."):
        # Display the summary
        for chunk in stqdm((chunks), desc= "Progress"):
            summary_placeholder = st.empty()
            try:
                chunk_summary = summarizer(chunk,  min_length = min_length, do_sample=False, no_repeat_ngram_size=4, encoder_no_repeat_ngram_size=3, num_beams=8, repetition_penalty=3.5) # max_length=150, min_length= 10 # temperature = temp, top_k = topk,  min_length=min_length, do_sample=False, no_repeat_ngram_size=ngram, encoder_no_repeat_ngram_size=3, num_beams=beams, early_stopping=False, repetition_penalty=3.5
                # Summarize chunk
                chunk_sum = chunk_summary[0]['summary_text']

                bulletedSummaryString += '\n⭕ ' + chunk_sum
                chunk_sum = "• " + chunk_sum
                # st.markdown(f'''• {chunk_sum}''')
                # Typewriting  
                for i in range(len(chunk_sum)+1):
                    summary_placeholder.markdown(chunk_sum[:i])
                    time.sleep(0.003)

            except Exception as e:
                print("Skipped chunk. Error:", e)    
        st.success("**Done Summarizing.**")
        return bulletedSummaryString
@st.cache_resource(show_spinner=False)
def find_summary_transformers(pdf_path, size_of_chunk, max_length, min_length): #, size_of_chunk, ngram, temp, topk, beams
    # Extract text using Tika
    with st.spinner('Parsing the document..'):
        document_text_part1 = extract_doc_text(pdf_path)
    
    subtab_tab_1, subtab_tab_2, subtab_tab_3 = st.tabs(['Summary','keywords','Document'])

    with subtab_tab_2:
        st.subheader('Top 10 Keywords')
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(document_text_part1)
        keyword_list= r.get_ranked_phrases()[:10]
        for keys in keyword_list:
            st.write("⭕", keys)

    with subtab_tab_3:
        st.subheader('Document')
        with st.expander('**View parsed data**', expanded=True):
            st.text(document_text_part1)
    
    with subtab_tab_1:
        st.subheader('Summary')
        global chunks
        with st.spinner("Creating chunks of data.."):
            chunks = text_chunking(document_text_part1, size_of_chunk) #, size_of_chunk
        if len(chunks) != 1:
            if len(chunks) <= 1000:
                all_transformers_summaries = transformers_summary(chunks, max_length, min_length) #, ngram, temp, topk, beams
                st.session_state.summary = all_transformers_summaries # added new logic
                summary_by_transformers = prep_b4_save(all_transformers_summaries)
                return summary_by_transformers
            else:
                st.write("Please upload a pdf with less than 500 pages!" )
        else:
            st.write("Not able to parse. Try another document!")
    


@st.cache_resource(show_spinner=False)
def get_pdf(input_string):
    try:
        
        # Create in-memory buffer
        pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

        lines = input_string.split('\n')
        elements = []
        styles = getSampleStyleSheet()
        styles['BodyText'].fontSize = 10
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                elements.append(Paragraph(stripped_line, styles['BodyText']))
                elements.append(Spacer(1, 10))

        def header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Times-Roman', 8)
            canvas.drawString(52,750,"AI Summarizer by Shivji" ) #AI Summarizer developed at  30
            canvas.drawString(453,50,"AI Summarizerby Shivji")
            canvas.restoreState()

        doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

        # Get the PDF data from the buffer and return it
        pdf_bytes = pdf_buffer.getvalue()
        return pdf_bytes 
    except Exception as e:
        return {"Exception is here: ": f'{e}'}


title = "AI Powered Python :grey[Summarization] Tool 📖"

# Create a title placeholder
title_placeholder =st.empty()
#**Select an Option**
choice = st.sidebar.radio(
    "🔻",
    ('File upload 📁', 'Website 🌐', 'Text Box 🖋️', 'Instructions ⚗️','Features 🦄', 'About 👩‍🦰')
)

# Typewriting  
for i in range(len(title)+1):
    title_placeholder.title(title[:i])
    time.sleep(0.003)

if choice == 'File upload 📁':
        uploaded_file = st.file_uploader(" ", type=["pdf", "docx", "txt"],  help="PDF, TXT and DOC file supported")
        st.write(" ")

        cola, colb = st.columns(2)
        with cola:
            min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
        with colb:
            size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250, help = 'Keep default value for best results, e.g. -250')
        if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)
        
        if st.button('Start Summarization'):
            if uploaded_file is not None:
                # file_details = {"Filename":uploaded_file.name, "FileSize":uploaded_file.size}
                if uploaded_file.name.endswith(('.pdf', '.docx', '.txt')):
                    # st.write(file_details)
                
                    # Display the summary
                    # st.subheader('Summary')
                    summary = find_summary_transformers(uploaded_file, size_of_chunk, max_length, min_length ) #, size_of_chunk, ngram, temp, topk, beams
                    st.session_state.summary = summary
                    st.subheader('Result')
                    pdf_bytes = get_pdf(summary)
                    b64 = base64.b64encode(pdf_bytes).decode()  # some strings
                    href = f'<a href="data:file/pdf;base64,{b64}" download="{uploaded_file.name}_summary.pdf">Download PDF File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.write("**Supported documents are PDF, Docx and txt!**")

elif choice == 'About 👩‍🦰':
    st.info("Designed and Developed by Shivji Agnihotri 🦄🤓 ")

elif choice == 'Text Box 🖋️':
    description = st.text_area('**Blog/ Article to summarize (min 150 words)**', placeholder= 'Once upon a time, in a small town nestled between rolling hills and lush greenery...')
    st.write("Adjust the parameters")
    
    cola, colb = st.columns(2)
    with cola:
        min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
    with colb:
        size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250,  help = 'Keep default value for best results, e.g. -250')

    if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)      
    
    if st.button('Start Summarization') and description and min_length and max_length: # and ngram:
        if len(description.split()) >= 150:
            if len(description.split()) < 50000:
                with st.spinner('Generating your summary..'):
                    

                    summary = transformers_summary(text_chunking(description, size_of_chunk), max_length, min_length) #, size_of_chunk , ngram, temp, topk, beams
                    st.session_state.summary = summary # added
                    st.subheader('Result')
                    pdf_bytes = get_pdf(summary)
                    b64 = base64.b64encode(pdf_bytes).decode()  # some strings
                    href = f'<a href="data:file/pdf;base64,{b64}" download="summary.pdf">Download PDF File</a>'
                    with st.expander("Download the pdf summary"):
                        st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("Your text exceeds the 50,000 words limit. Please shorten your text.")
        else:
            st.write("**Too short to Summarize!**")

elif choice == 'Website 🌐':
    from scraper.getSerpResults import scrape_content
    url = st.text_input("**Paste a URL**", placeholder=" Paste a website's url..")
    st.write("Adjust the parameters")

    cola, colb = st.columns(2)
    with cola:
        min_length, max_length = st.slider( 'Select a range of Summarization', 15, 50, (33, 39))
    with colb:
        size_of_chunk = st.slider( 'Adjust Chunk Size', 250, 1000, value=250, help = 'Keep default value for best results, e.g. -250')

    if st.session_state.summary is not None:
            with st.expander("Your past summary:"):
                st.text(st.session_state.summary)

    title, text, summary, keywords = scrape_content(url)
    if st.button('Start Summarization') and text:
        if len(text.split()) >= 50:
            if len(text.split()) < 50000:
                
                st.divider()
                # Display the title
                st.markdown(f"## {title}")
                
                # Display the summary
                st.subheader("Summary")
                with st.spinner():
                    summary = transformers_summary(text_chunking(text, size_of_chunk), max_length, min_length) #, size_of_chunk, ngram, temp, topk, beams
                st.session_state.summary = summary
                st.subheader('Result')
                with st.expander("Download the pdf summary"):
                    pdf_bytes = get_pdf(summary)
                    b64 = base64.b64encode(pdf_bytes).decode()  # some strings
                    href = f'<a href="data:file/pdf;base64,{b64}" download="Summary.pdf">Download PDF File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                st.divider() 

                with st.expander('**View page keywords**'):
                    # Display the keywords
                    st.markdown(f"**Keywords:**\n{', '.join(keywords)}")
                
                with st.expander('**View page text**'):
                    # Display the text
                    st.markdown("**Text:**")
                    st.text(text)
            else:
                st.write("Your text exceeds the 50,000 words limit. Please shorten your text.")
        else:
            st.write("**Too short to Summarize!**")

# st.divider()   
elif choice == 'Instructions ⚗️':
    st.info("To enhance the performance of a summarization model, you can fine-tune several parameters:")

    st.markdown("""
    - **Min Length**: Defines the minimum length of the generated text. It ensures that the output is not too short and contains enough information.
    - **Max Length**: Sets an upper limit to the length of the generated text. It prevents the output from being excessively long.
    - **Chunk Size**: The portion of text in words to summarize at once. It affects the depth of summary getting generated.
    """)

elif choice == 'Features 🦄':
        st.markdown('''### Here's what makes my tool stand out:\n
        - 🚀 **Fast**: Get your summaries in seconds.\n
        - 🧠 **Smart**: Uses AI to understand context and extract key points.\n
        - 📚 **Versatile**: Perfect for academic papers, reports, books, and more.\n
        - 🔒 **Secure**: Your documents are safe with us. We respect your privacy.''')
