import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import torch
import re
from typing import List, Dict
from database import Database
from datetime import datetime
from langdetect import detect
from deep_translator import GoogleTranslator
from datetime import datetime
from streamlit_js_eval import streamlit_js_eval

def set_cookie(name, value):
    streamlit_js_eval(js_expressions=f"document.cookie = '{name}={value}; path=/'")

def get_cookie(name):
    cookie_str = streamlit_js_eval(js_expressions="document.cookie")
    cookies = dict(item.split("=") for item in cookie_str.split("; ") if "=" in item)
    return cookies.get(name)

# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

# Initialize database
db = Database()

# Page config
st.set_page_config(
    page_title="Indian Law Q&A Assistant",
    page_icon="⚖️",
    layout="wide"
)


st.markdown(
    r"""
    <style>
        .stAppDeployButton {display:none;}
        .stDeployButton {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom prompt template
PROMPT_TEMPLATE = """<|im_start|>system
You are a legal assistant specializing in Indian Law and court jurisdictions. Your role is to:
1. Analyze case details and determine which court has jurisdiction
2. Explain the reasoning behind the court selection
3. Provide relevant legal sections or precedents from the context
4. If a case could fall under multiple jurisdictions, explain each possibility

When analyzing jurisdiction:
- For civil matters: Consider the case value, geographical location, and subject matter
- For criminal matters: Consider the offense severity and applicable IPC sections
- For consumer cases: Check if it's about goods/services and the claim amount
- For family matters: Look for matrimonial, inheritance, or custody issues
- For constitutional matters: Check if fundamental rights or constitutional issues are involved

If the provided case details are insufficient, specify what additional information is needed to determine jurisdiction accurately.
<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}
<|im_end|>
<|im_start|>assistant
Based on the provided case details and legal context, let me analyze the jurisdiction:
"""

@st.cache_resource
def initialize_llm():
    """Initialize the language model"""
    try:
        # Using a smaller model that works well on CPU
        repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load the model with CPU configuration
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Create pipeline with proper chat formatting
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,  # Reduced temperature for more focused responses
            top_p=0.85,      # Adjusted for better coherence
            repetition_penalty=1.2,
            # device="cpu",
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            return_full_text=False
        )
        
        # Create LangChain wrapper
        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"temperature": 0.3}
        )
        return llm
        
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

@st.cache_resource
def initialize_qa_chain():
    """Initialize the QA chain with embeddings and vector store"""
    try:
        # Initialize LLM
        llm = initialize_llm()
        if llm is None:
            st.error("Failed to initialize LLM")
            return None
            
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        if os.path.exists("vectorstore"):
            vectorstore = Chroma(
                persist_directory="vectorstore",
                embedding_function=embeddings
            )
        else:
            st.error("Vector store not found. Please run the ingestion script first.")
            return None
            
        # Create prompt
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Initialize QA chain with simpler retrieval configuration
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 4  # Number of documents to retrieve
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            }
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing the system: {str(e)}")
        return None

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting"""
    greetings = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
                 'good evening', 'hi there', 'hello there', 'namaste'}
    return text.lower().strip() in greetings

def handle_greeting() -> str:
    """Return a welcoming greeting message"""
    return ("Hello! I'm your Indian Law Assistant. Please ask your questions about Indian law, "
            "and I'll provide accurate information based on legal documents.")

def is_jurisdiction_query(text: str) -> bool:
    """Check if the query is about court jurisdiction"""
    keywords = {
        'which court', 'court jurisdiction', 'where to file', 'which jurisdiction',
        'civil court', 'criminal court', 'consumer court', 'high court', 'supreme court',
        'jurisdiction', 'file case', 'file complaint', 'where should i go', 'which court should'
    }
    translator = GoogleTranslator(source='auto', target='en')
    check_lang = translator.detect(text)
    if check_lang == 'ml':
        text = translator.translate(text)
    return any(keyword in text.lower() for keyword in keywords)

def extract_case_details(text: str) -> dict:
    """Extract available case details from the query"""
    details = {
        'nature': None,  # civil, criminal, consumer, etc.
        'value': None,   # monetary value
        'location': None,  # geographical location
        'issue': None,    # main complaint/issue
        'parties': None   # involved parties
    }
    
    # Check for case nature
    text_lower = text.lower()
    
    # Criminal cases
    if any(word in text_lower for word in ['hit and run', 'accident', 'hit a car', 'vehicle accident', 
                                         'criminal', 'theft', 'assault', 'fraud', 'cheating']):
        details['nature'] = 'criminal'
        details['issue'] = 'Vehicle accident/Criminal matter'
    
    # Civil cases
    elif any(word in text_lower for word in ['property', 'tenant', 'rent', 'contract', 'agreement', 
                                           'civil', 'damage', 'compensation']):
        details['nature'] = 'civil'
        details['issue'] = 'Civil dispute'
    
    # Consumer cases
    elif any(word in text_lower for word in ['defective', 'product', 'service', 'consumer', 
                                           'refund', 'warranty', 'deficiency']):
        details['nature'] = 'consumer'
        details['issue'] = 'Consumer dispute'
    
    # Family cases
    elif any(word in text_lower for word in ['divorce', 'marriage', 'custody', 'maintenance', 
                                           'family', 'matrimonial']):
        details['nature'] = 'family'
        details['issue'] = 'Family matter'
    
    # If no specific nature is found but it's a jurisdiction query, default to criminal
    elif is_jurisdiction_query(text):
        details['nature'] = 'criminal'
        details['issue'] = 'General criminal matter'
    
    # Extract monetary values
    value_patterns = [
        r'(?:rs\.?|inr|rupees?)\s*(\d+(?:,\d+)*(?:\.\d{2})?)',  # Rs. 20,000
        r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rs\.?|inr|rupees?)',  # 20,000 Rs
        r'(\d+)k',  # 20k format
        r'(\d+)\s*thousand',  # 20 thousand format
        r'(\d+)\s*lakh',  # 2 lakh format
        r'(\d+)\s*crore'  # 1 crore format
    ]
    
    for pattern in value_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            details['value'] = matches[0]
            break
    
    return details

def get_jurisdiction_prompt(query: str, conversation_history: List[Dict] = None) -> str:
    """Get smart prompts for jurisdiction-related queries based on available information"""
    # First, analyze the current query
    details = extract_case_details(query)
    
    # If we have conversation history, analyze it too
    if conversation_history:
        conv_context = get_conversation_context(conversation_history)
        context_details = extract_case_details(conv_context)
        
        # Merge details, preferring current query details over context
        for key, value in context_details.items():
            if details[key] is None:
                details[key] = value
    
    # Build a list of missing important details
    missing_details = []
    
    if not details['nature']:
        missing_details.append("What is the nature of the case? (civil, criminal, consumer dispute, family matter, etc.)")
    
    if not details['value'] and details['nature'] in ['civil', 'consumer']:
        missing_details.append("What is the approximate value of the dispute?")
    
    if not details['location']:
        missing_details.append("Where are you located?")
    
    if not details['issue'] and len(query.split()) < 10:  # Only ask for issue if query is short and no issue detected
        missing_details.append("What is the main issue or complaint?")
    
    # Only ask for parties if we don't have enough context
    if not details['parties'] and len(missing_details) > 0:
        missing_details.append("Who are the parties involved (without personal details)?")
    
    if missing_details:
        response = "To determine the appropriate court jurisdiction, please provide these additional details:\n\n"
        response += "\n".join(f"{i+1}. {detail}" for i, detail in enumerate(missing_details))
        return response
    
    return None

def get_conversation_context(messages: List[Dict]) -> str:
    """Build context from previous messages"""
    if len(messages) < 2:  # No previous context
        return ""
    
    # Get last 3 exchanges (up to 6 messages)
    relevant_messages = messages[-6:]
    context = []
    
    for msg in relevant_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        context.append(f"{role}: {content}")
    
    return "\n".join(context)

def is_followup_question(text: str) -> bool:
    """Check if the question is a follow-up"""
    followup_indicators = {
        'what about', 'and what', 'then what', 'in that case',
        'so', 'therefore', 'in this case', 'for this', 'this',
        'that', 'those', 'these', 'they', 'it', 'related to',
        'following up', 'additionally', 'moreover', 'further',
        'also', 'another question', 'one more thing'
    }
    
    # Check for pronouns and demonstratives
    has_pronouns = any(word in text.lower().split() for word in ['it', 'this', 'that', 'these', 'those', 'they'])
    
    # Check for follow-up phrases
    has_followup_phrase = any(phrase in text.lower() for phrase in followup_indicators)
    
    return has_pronouns or has_followup_phrase

def convert_indian_number(value_str: str) -> float:
    """Convert Indian number format (with commas) to float"""
    if not value_str:
        return 0.0
    # Remove commas and convert to float
    clean_value = value_str.replace(',', '')
    try:
        return float(clean_value)
    except ValueError:
        return 0.0

def detect_language(text: str) -> str:
    """Detect if text is in Malayalam or English"""
    try:
        # First check for Malayalam characters
        malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')
        if malayalam_pattern.search(text):
            return 'ml'
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def translate_response(text: str, target_lang: str) -> str:
    """Translate text to target language"""
    try:
        if target_lang == 'ml':
            translator = GoogleTranslator(source='en', target='ml')
            return translator.translate(text)
        elif target_lang == 'en':
            translator = GoogleTranslator(source='ml', target='en')
            return translator.translate(text)
        return text
    except:
        return text  # Return original text if translation fails

def determine_court_jurisdiction(case_details: dict) -> Dict:
    """Determine appropriate court jurisdiction based on case details"""
    jurisdiction = {
        "primary_court": None,
        "reason": "",
        "alternative_courts": []
    }
    
    nature = case_details.get('nature')
    value_str = case_details.get('value')
    value = convert_indian_number(value_str) if value_str else None
    issue = case_details.get('issue', '')
    
    # Get language of the issue
    lang = detect_language(issue)
    
    if nature == 'criminal':
        if 'accident' in issue.lower() or 'hit' in issue.lower():
            jurisdiction["primary_court"] = "Chief Metropolitan Magistrate Court"
            reason = "Vehicle accidents and hit-and-run cases are typically handled by the Chief Metropolitan Magistrate Court. The court will determine the severity of the offense and appropriate legal action."
        else:
            jurisdiction["primary_court"] = "Metropolitan Magistrate Court"
            reason = "Criminal matters are typically handled by Metropolitan Magistrate Courts"
        
        # Translate reason if issue was in Malayalam
        if lang == 'ml':
            mal_reason = translate_response(reason, 'ml')
            jurisdiction["reason"] = f"{mal_reason}\n\nEnglish:\n{reason}"
        else:
            jurisdiction["reason"] = reason
            
    elif nature == 'civil':
        if value and value <= 100000:  # 1 lakh
            jurisdiction["primary_court"] = "City Civil Court"
            reason = "Civil matters up to Rs. 1 lakh fall under City Civil Court jurisdiction"
        else:
            jurisdiction["primary_court"] = "District Court"
            reason = "Civil matters above Rs. 1 lakh fall under District Court jurisdiction"
            
        if lang == 'ml':
            mal_reason = translate_response(reason, 'ml')
            jurisdiction["reason"] = f"{mal_reason}\n\nEnglish:\n{reason}"
        else:
            jurisdiction["reason"] = reason
    
    return jurisdiction

def handle_legal_query(query: str, qa_chain, conversation_history: List[Dict]) -> str:
    """Handle legal questions using the QA chain"""
    try:
        # Detect language
        lang = detect_language(query)
        
        # If query is in Malayalam, translate to English for processing
        if lang == 'ml':
            eng_query = translate_response(query, 'en')
        else:
            eng_query = query
        
        # Get conversation context
        context = get_conversation_context(conversation_history)
        
        # If it's a follow-up, include previous context
        if is_followup_question(eng_query) and context:
            query_with_context = f"Previous context: {context}\nQuestion: {eng_query}\nPlease give a simple, 2-3 line answer that anyone can understand."
        else:
            query_with_context = f"Question: {eng_query}\nPlease give a simple, 2-3 line answer that anyone can understand."
        
        # Get response from QA chain
        result = qa_chain({"query": query_with_context})
        response = result.get('result', "I'm sorry, I couldn't understand that. Could you ask in simpler words?")
        
        # Clean up the response
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        response = response.replace("[INST]", "").replace("[/INST]", "")
        response = response.strip()
        
        # Ensure response is concise
        sentences = response.split('.')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        # If original query was in Malayalam, translate response to Malayalam
        if lang == 'ml':
            mal_response = translate_response(response, 'ml')
            response = f"{mal_response}\n\nEnglish translation:\n{response}"
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        error_msg = "ക്ഷമിക്കണം, ദയവായി വീണ്ടും ചോദിക്കാമോ?\n\nSorry, could you ask that again?"
        return error_msg

def get_courts_by_location_and_type(location: str, case_type: str) -> Dict[str, List[str]]:
    """Get available courts based on location and case type"""
    courts = {}
    
    # Define courts by location
    location_courts = {
        'Chennai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Egmore',
                    'Metropolitan Magistrate Court, George Town',
                    'Metropolitan Magistrate Court, Saidapet'
                ],
                'Sessions Courts': [
                    'Principal Sessions Court, Chennai',
                    'Additional Sessions Court, Chennai',
                    'Special Court for CBI Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Chennai',
                    'Small Causes Court, Chennai',
                    'District Court, Chennai'
                ],
                'High Court': [
                    'Madras High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Chennai',
                    'State Consumer Disputes Redressal Commission, Chennai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Chennai',
                    'Additional Family Court, Chennai'
                ]
            }
        },
        'Mumbai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Fort',
                    'Metropolitan Magistrate Court, Bandra',
                    'Metropolitan Magistrate Court, Andheri'
                ],
                'Sessions Courts': [
                    'Sessions Court, Mumbai',
                    'Special Court for NDPS Cases',
                    'Special CBI Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Mumbai',
                    'Small Causes Court, Mumbai',
                    'District Court, Mumbai'
                ],
                'High Court': [
                    'Bombay High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum, Mumbai',
                    'State Consumer Commission, Mumbai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Mumbai',
                    'Additional Family Court, Bandra'
                ]
            }
        },
        'Delhi': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court',
                    'Metropolitan Magistrate Court',
                    'Judicial Magistrate First Class'
                ],
                'Sessions Courts': [
                    'Sessions Court',
                    'Special Criminal Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court',
                    'Small Causes Court',
                    'City Civil Court'
                ],
                'High Court': [
                    'Delhi High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum',
                    'State Consumer Commission'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Delhi',
                    'Additional Family Court'
                ]
            }
        },
        'Kerala': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Judicial Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Thiruvananthapuram'
                ],
                'Sessions Courts': [
                    'Sessions Court, Ernakulam',
                    'Sessions Court, Thiruvananthapuram',
                    'Special Court for NDPS Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court, Ernakulam',
                    'District Court, Thiruvananthapuram',
                    'District Court, Kozhikode'
                ],
                'High Court': [
                    'High Court of Kerala, Ernakulam'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Ernakulam',
                    'State Consumer Disputes Redressal Commission, Ernakulam'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Ernakulam',
                    'Family Court, Thiruvananthapuram',
                    'Family Court, Kozhikode'
                ]
            }
        }
    }
    
    # Get courts for the specified location and case type
    if location in location_courts and case_type:
        case_type = case_type.lower()
        if case_type in location_courts[location]:
            return location_courts[location][case_type]
    
    return courts
def get_advocates_by_location(location: str) -> List[Dict]:
    """Get available advocates based on location"""
    advocates_by_location = {
        'Chennai': {
            'criminal': [
                {
                    'name': 'Adv. Rajesh Kumar',
                    'specialization': 'Criminal Defense',
                    'experience': '15 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Lakshmi Narayan',
                    'specialization': 'Criminal Law',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Priya Raman',
                    'specialization': 'Civil Litigation',
                    'experience': '12 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Senthil Kumar',
                    'specialization': 'Property Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'consumer': [
                {
                    'name': 'Adv. Meena Krishnan',
                    'specialization': 'Consumer Protection',
                    'experience': '10 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'family': [
                {
                    'name': 'Adv. Sudha Raghavan',
                    'specialization': 'Family Law',
                    'experience': '16 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Mumbai': {
            'criminal': [
                {
                    'name': 'Adv. Prakash Shah',
                    'specialization': 'Criminal Defense',
                    'experience': '22 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Anjali Desai',
                    'specialization': 'Civil Law',
                    'experience': '15 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Delhi': {
            'criminal': [
                {
                    'name': 'Adv. Mohammed Ali',
                    'specialization': 'Criminal Defense',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Sarah Thomas',
                    'specialization': 'Criminal Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Kerala': {
            'criminal': [
                {
                    'name': 'Adv. Suresh Kumar',
                    'specialization': 'Criminal Defense',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Ramesh Nair',
                    'specialization': 'Criminal Law',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Sreedevi Pillai',
                    'specialization': 'Civil Litigation',
                    'experience': '12 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Sreekumar Menon',
                    'specialization': 'Property Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'consumer': [
                {
                    'name': 'Adv. Sreekala Sreedharan',
                    'specialization': 'Consumer Protection',
                    'experience': '10 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'family': [
                {
                    'name': 'Adv. Sreedevi Sreedharan',
                    'specialization': 'Family Law',
                    'experience': '16 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        }
    }
    
    if location in advocates_by_location:
        return advocates_by_location[location]
    
    return []
def get_advocates_by_specialization(court_type: str, case_type: str, location: str) -> List[Dict]:
    """Get available advocates based on specialization and court type"""
    advocates_by_location = {
        'Chennai': {
            'criminal': [
                {
                    'name': 'Adv. Rajesh Kumar',
                    'specialization': 'Criminal Defense',
                    'experience': '15 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Lakshmi Narayan',
                    'specialization': 'Criminal Law',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Priya Raman',
                    'specialization': 'Civil Litigation',
                    'experience': '12 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Senthil Kumar',
                    'specialization': 'Property Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'consumer': [
                {
                    'name': 'Adv. Meena Krishnan',
                    'specialization': 'Consumer Protection',
                    'experience': '10 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'family': [
                {
                    'name': 'Adv. Sudha Raghavan',
                    'specialization': 'Family Law',
                    'experience': '16 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Mumbai': {
            'criminal': [
                {
                    'name': 'Adv. Prakash Shah',
                    'specialization': 'Criminal Defense',
                    'experience': '22 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Anjali Desai',
                    'specialization': 'Civil Law',
                    'experience': '15 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Delhi': {
            'criminal': [
                {
                    'name': 'Adv. Mohammed Ali',
                    'specialization': 'Criminal Defense',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Sarah Thomas',
                    'specialization': 'Criminal Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        },
        'Kerala': {
            'criminal': [
                {
                    'name': 'Adv. Suresh Kumar',
                    'specialization': 'Criminal Defense',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Ramesh Nair',
                    'specialization': 'Criminal Law',
                    'experience': '20 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'civil': [
                {
                    'name': 'Adv. Sreedevi Pillai',
                    'specialization': 'Civil Litigation',
                    'experience': '12 years',
                    'contact': '+91-XXXXXXXXXX'
                },
                {
                    'name': 'Adv. Sreekumar Menon',
                    'specialization': 'Property Law',
                    'experience': '18 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'consumer': [
                {
                    'name': 'Adv. Sreekala Sreedharan',
                    'specialization': 'Consumer Protection',
                    'experience': '10 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ],
            'family': [
                {
                    'name': 'Adv. Sreedevi Sreedharan',
                    'specialization': 'Family Law',
                    'experience': '16 years',
                    'contact': '+91-XXXXXXXXXX'
                }
            ]
        }
    }
    
    if location in advocates_by_location and case_type:
        case_type = case_type.lower()
        return advocates_by_location[location].get(case_type, [])
    
    return []

def init_session_state():
    """Initialize session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'

def load_user_chat_history():
    """Load chat history for the current user"""
    if st.session_state.user:
        messages = db.get_user_chat_history(st.session_state.user['id'])
        st.session_state.messages = messages

def get_courts_by_location_and_type(location: str, case_type: str) -> Dict[str, List[str]]:
    """Get available courts based on location and case type"""
    courts = {}
    
    # Define courts by location
    location_courts = {
        'Chennai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Egmore',
                    'Metropolitan Magistrate Court, George Town',
                    'Metropolitan Magistrate Court, Saidapet'
                ],
                'Sessions Courts': [
                    'Principal Sessions Court, Chennai',
                    'Additional Sessions Court, Chennai',
                    'Special Court for CBI Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Chennai',
                    'Small Causes Court, Chennai',
                    'District Court, Chennai'
                ],
                'High Court': [
                    'Madras High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Chennai',
                    'State Consumer Disputes Redressal Commission, Chennai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Chennai',
                    'Additional Family Court, Chennai'
                ]
            }
        },
        'Mumbai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Fort',
                    'Metropolitan Magistrate Court, Bandra',
                    'Metropolitan Magistrate Court, Andheri'
                ],
                'Sessions Courts': [
                    'Sessions Court, Mumbai',
                    'Special Court for NDPS Cases',
                    'Special CBI Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Mumbai',
                    'Small Causes Court, Mumbai',
                    'District Court, Mumbai'
                ],
                'High Court': [
                    'Bombay High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum, Mumbai',
                    'State Consumer Commission, Mumbai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Mumbai',
                    'Additional Family Court, Bandra'
                ]
            }
        },
        'Delhi': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court',
                    'Metropolitan Magistrate Court',
                    'Judicial Magistrate First Class'
                ],
                'Sessions Courts': [
                    'Sessions Court',
                    'Special Criminal Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court',
                    'Small Causes Court',
                    'City Civil Court'
                ],
                'High Court': [
                    'Delhi High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum',
                    'State Consumer Commission'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Delhi',
                    'Additional Family Court'
                ]
            }
        },
        'Kerala': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Judicial Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Thiruvananthapuram'
                ],
                'Sessions Courts': [
                    'Sessions Court, Ernakulam',
                    'Sessions Court, Thiruvananthapuram',
                    'Special Court for NDPS Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court, Ernakulam',
                    'District Court, Thiruvananthapuram',
                    'District Court, Kozhikode'
                ],
                'High Court': [
                    'High Court of Kerala, Ernakulam'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Ernakulam',
                    'State Consumer Disputes Redressal Commission, Ernakulam'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Ernakulam',
                    'Family Court, Thiruvananthapuram',
                    'Family Court, Kozhikode'
                ]
            }
        }
    }
    
    # Get courts for the specified location and case type
    if location in location_courts and case_type:
        case_type = case_type.lower()
        if case_type in location_courts[location]:
            return location_courts[location][case_type]
    
    return courts

def get_courts_by_location(location: str) -> Dict[str, List[str]]:
    """Get available courts based on location and case type"""
    courts = {}
    
    # Define courts by location
    location_courts = {
        'Chennai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Egmore',
                    'Metropolitan Magistrate Court, George Town',
                    'Metropolitan Magistrate Court, Saidapet'
                ],
                'Sessions Courts': [
                    'Principal Sessions Court, Chennai',
                    'Additional Sessions Court, Chennai',
                    'Special Court for CBI Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Chennai',
                    'Small Causes Court, Chennai',
                    'District Court, Chennai'
                ],
                'High Court': [
                    'Madras High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Chennai',
                    'State Consumer Disputes Redressal Commission, Chennai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Chennai',
                    'Additional Family Court, Chennai'
                ]
            }
        },
        'Mumbai': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court, Fort',
                    'Metropolitan Magistrate Court, Bandra',
                    'Metropolitan Magistrate Court, Andheri'
                ],
                'Sessions Courts': [
                    'Sessions Court, Mumbai',
                    'Special Court for NDPS Cases',
                    'Special CBI Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'City Civil Court, Mumbai',
                    'Small Causes Court, Mumbai',
                    'District Court, Mumbai'
                ],
                'High Court': [
                    'Bombay High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum, Mumbai',
                    'State Consumer Commission, Mumbai'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Mumbai',
                    'Additional Family Court, Bandra'
                ]
            }
        },
        'Delhi': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Metropolitan Magistrate Court',
                    'Metropolitan Magistrate Court',
                    'Judicial Magistrate First Class'
                ],
                'Sessions Courts': [
                    'Sessions Court',
                    'Special Criminal Court'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court',
                    'Small Causes Court',
                    'City Civil Court'
                ],
                'High Court': [
                    'Delhi High Court'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Forum',
                    'State Consumer Commission'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Delhi',
                    'Additional Family Court'
                ]
            }
        },
        'Kerala': {
            'criminal': {
                'Magistrate Courts': [
                    'Chief Judicial Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Ernakulam',
                    'Judicial First Class Magistrate Court, Thiruvananthapuram'
                ],
                'Sessions Courts': [
                    'Sessions Court, Ernakulam',
                    'Sessions Court, Thiruvananthapuram',
                    'Special Court for NDPS Cases'
                ]
            },
            'civil': {
                'Civil Courts': [
                    'District Court, Ernakulam',
                    'District Court, Thiruvananthapuram',
                    'District Court, Kozhikode'
                ],
                'High Court': [
                    'High Court of Kerala, Ernakulam'
                ]
            },
            'consumer': {
                'Consumer Courts': [
                    'District Consumer Disputes Redressal Forum, Ernakulam',
                    'State Consumer Disputes Redressal Commission, Ernakulam'
                ]
            },
            'family': {
                'Family Courts': [
                    'Family Court, Ernakulam',
                    'Family Court, Thiruvananthapuram',
                    'Family Court, Kozhikode'
                ]
            }
        }
    }
    
    # Get courts for the specified location and case type
    if location in location_courts:
        return location_courts[location]
    
    return courts


def show_login_page():
    """Display the login page"""
    st.title("Login to Legal Assistant")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username and password:
                user = db.login_user(username, password)
                if user:
                    set_cookie("user_id", user['id'])
                    st.session_state.user = user
                    st.session_state.current_page = 'main'
                    load_user_chat_history()  # Load user's chat history
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please fill in all fields")
    
    if st.button("Don't have an account? Register here"):
        st.session_state.current_page = 'register'
        st.rerun()

def show_register_page():
    """Display the registration page"""
    st.title("Register for Legal Assistant")
    
    with st.form("register_form"):
        username = st.text_input("Username*")
        email = st.text_input("Email*")
        password = st.text_input("Password*", type="password")
        confirm_password = st.text_input("Confirm Password*", type="password")
        full_name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        location = st.selectbox(
            "Location",
            ["", "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Kerala"]
        )
        
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                if db.register_user(username, email, password, full_name, phone, location):
                    st.success("Registration successful! Please login.")
                    st.session_state.current_page = 'login'
                    st.rerun()
                else:
                    st.error("Username or email already exists")
    
    if st.button("Already have an account? Login here"):
        st.session_state.current_page = 'login'
        st.rerun()

def show_profile_page():
    """Display the user profile page"""
    st.title("User Profile")
    
    user = db.get_user_profile(st.session_state.user['id'])
    if not user:
        st.error("Error loading profile")
        return
    
    with st.form("profile_form"):
        email = st.text_input("Email", value=user['email'])
        full_name = st.text_input("Full Name", value=user['full_name'] or "")
        phone = st.text_input("Phone Number", value=user['phone'] or "")
        location = st.selectbox(
            "Location",
            ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Kerala"],
            index=["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Kerala"].index(user['location']) if user['location'] else 0
        )
        
        if st.form_submit_button("Update Profile"):
            updates = {
                'email': email,
                'full_name': full_name,
                'phone': phone,
                'location': location
            }
            if db.update_user_profile(user['id'], updates):
                st.success("Profile updated successfully!")
                st.session_state.user = db.get_user_profile(user['id'])
            else:
                st.error("Error updating profile")
    
    if st.button("Back to Main Page"):
        st.session_state.current_page = 'main'
        st.rerun()

def show_sidebar():
    """Display the sidebar with user information and navigation"""
    with st.sidebar:
        if st.session_state.user:
            st.write(f"Welcome, {st.session_state.user['username']}!")
            if st.button("Home"):
                st.session_state.user = None
                st.session_state.current_page = 'main'
                st.rerun()
            if st.button("Interactive Q & A"):
                st.session_state.current_page = 'interactive_qa'
                st.rerun()
            if st.button("View Profile"):
                st.session_state.current_page = 'profile'
                st.rerun()
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.current_page = 'login'
                st.rerun()

def show_interactive_qa():
    """Display the main chat interface"""
    st.title("Indian Law Q&A Assistant ")
    
    # Location selection in sidebar
    user_location = st.session_state.user.get('location', 'Kerala')
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()
    if qa_chain is None:
        st.error("System initialization failed. Please check the logs.")
        return
    
    # Add clear chat history button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear Chat"):
            if db.clear_user_chat_history(st.session_state.user['id']):
                st.session_state.messages = []
                st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your legal question..."):
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        db.save_chat_message(st.session_state.user['id'], "user", prompt)
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with loading animation
        with st.chat_message("assistant"):
            with st.spinner("ചിന്തിക്കുന്നു... Thinking..."):
                if is_greeting(prompt):
                    response = handle_greeting()
                elif is_jurisdiction_query(prompt):
                    case_details = extract_case_details(prompt)
                    jurisdiction = determine_court_jurisdiction(case_details)
                    
                    response = f"You should go to: **{jurisdiction['primary_court']}**\n\n"
                    response += f"Why? {jurisdiction['reason']}\n\n"
                    
                    courts = get_courts_by_location_and_type(user_location, case_details.get('nature'))
                    
                    if courts:
                        response += f"Here are the nearby courts in {user_location}:\n"
                        for court_category, court_list in courts.items():
                            response += f"\n*{court_category}:*\n"
                            for court in court_list[:2]:  # Show only top 2 courts
                                response += f"- {court}\n"
                        
                        advocates = get_advocates_by_specialization(
                            jurisdiction['primary_court'],
                            case_details.get('nature'),
                            user_location
                        )
                        
                        if advocates:
                            response += f"\nHere are 2 lawyers who can help you:\n"
                            for advocate in advocates[:2]:  # Show only top 2 advocates
                                response += f"- {advocate['name']} ({advocate['specialization']}, {advocate['experience']})\n  Contact: {advocate['contact']}\n"
                else:
                    response = handle_legal_query(prompt, qa_chain, st.session_state.messages)
                
                # Save and display assistant response
                db.save_chat_message(st.session_state.user['id'], "assistant", response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
def show_main_page():
     # Custom CSS
    st.markdown("""
        <style>
            .hero-section {
                background: linear-gradient(90deg, #1a237e 0%, #283593 100%);
                padding: 4rem 2rem;
                color: white;
                text-align: center;
                border-radius: 0px 0px 20px 20px;
                margin-bottom: 2rem;
            }
            .service-box {
                background-color: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                height: 100%;
                min-height: 300px;
                cursor: pointer;
                transition: transform 0.2s;
                text-align: center;
            }
            .service-box:hover {
                transform: translateY(-5px);
            }
            .icon-large {
                font-size: 3rem;
                margin-bottom: 1rem;
                color: #1a237e;
            }
            .qa-input {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 style='font-size: 3.5rem; font-weight: bold; margin-bottom: 1rem;'>
                Legal Services Portal
            </h1>
            <p style='font-size: 1.5rem; opacity: 0.9; margin-bottom: 2rem;'>
                Your gateway to legal resources and assistance
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for QA history
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    
    # Main Services Section
    col1, col2, col3 = st.columns(3)
    
    # Box 1: Interactive Q&A
    with col1:
        st.markdown("""
            <div class="service-box">
                <div class="icon-large">❓</div>
                <h3 style='color: #1a237e; font-size: 1.5rem; margin-bottom: 1rem;'>
                    Legal Q&A Assistant
                </h3>
            """, unsafe_allow_html=True)
        
        # Create a container for centered content
        col1_center = st.container()
        with col1_center:
            # Create three columns to center the button
            left_spacer, center_content, right_spacer = st.columns([1, 2, 1])
            with center_content:
                if st.button("Ask Q&A", key="ask_qa", use_container_width=True):
                    st.session_state.current_page = "interactive_qa"
                    st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Box 2: Nearby Courts Navigation Card
    with col2:
        st.markdown("""
            <div class="service-box">
                <div class="icon-large">⚖️</div>
                <h3 style='color: #1a237e; font-size: 1.5rem; margin-bottom: 1rem;'>
                    Nearby Courts
                </h3>
                <p style='color: #666; margin-bottom: 2rem;'>
                    Find courts in your vicinity with detailed information about their jurisdiction and services.
                </p>
            """, unsafe_allow_html=True)
        
        # Create a container for centered content
        col1_center = st.container()
        with col1_center:
            # Create three columns to center the button
            left_spacer, center_content, right_spacer = st.columns([1, 2, 1])
            with center_content:
                if st.button("Nearby Courts", key="nearby_courts",use_container_width=True):
                    st.session_state.current_page = "nearby_courts"
                    st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Box 3: Nearby Advocates Navigation Card
    with col3:
        st.markdown("""
            <div class="service-box">
                <div class="icon-large">👨‍⚖️</div>
                <h3 style='color: #1a237e; font-size: 1.5rem; margin-bottom: 1rem;'>
                    Nearby Advocates
                </h3>
                <p style='color: #666; margin-bottom: 2rem;'>
                    Connect with experienced legal professionals in your area specializing in various practice areas.
                </p>
            """, unsafe_allow_html=True)
        
        # Create a container for centered content
        col1_center = st.container()
        with col1_center:
            # Create three columns to center the button
            left_spacer, center_content, right_spacer = st.columns([1, 2, 1])
            with center_content:
                if st.button("Nearby Advocates", key="nearby_advocates",use_container_width=True):
                    st.session_state.current_page = "nearby_advocates"
                    st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

def show_nearby_courts():
    """Display nearby courts based on user location"""
    st.title("Nearby Courts")

    # Get user's location
    user_location = st.session_state.user.get('location', 'Kerala')
    courts = get_courts_by_location(user_location)

    if courts:
        # Add introductory text
        st.markdown(f"""
            <div style='margin-bottom: 20px;'>
                <p>Showing courts near <strong>{user_location}</strong>.</p>
            </div>
        """, unsafe_allow_html=True)

        # Display courts by category
        for court_category, court_list in courts.items():
            st.subheader(f"{court_category}")

            # Create columns for better layout
            cols = st.columns(2)
            for i, (court, details) in enumerate(court_list.items()):  # Assume `court_list` is a dictionary
                with cols[i % 2]:
                    # Render court name with details as bullet points
                    st.markdown(f"""
                        <div style='padding: 15px; border: 1px solid #e0e0e0; border-radius: 5px; margin-bottom: 10px;'>
                            <h4 style='margin: 0; color: #1a237e;'>{court}</h4>
                            <ul style='margin: 10px 0 0 20px; color: #444;'>
                                {"".join(f"<li>{detail}</li>" for detail in details)}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info(f"No courts found in {user_location}. Please try a different location.")

def show_nearby_advocates():
    """Display nearby advocates based on user location"""
    st.title("Nearby Advocates")
    
    # Get user's location
    user_location = st.session_state.user.get('location', 'Kerala')
    advocates = get_advocates_by_location(user_location)
    
    if advocates:
        # Add introductory text
        st.markdown(f"""
            <div style='margin-bottom: 20px;'>
                <p>Showing advocates near <strong>{user_location}</strong>. 
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display advocates by practice area
        for court_type, advocate_list in advocates.items():
            st.subheader(f"{court_type.title()} Law Specialists")
            
            # Create columns for better layout
            cols = st.columns(2)
            for i, advocate in enumerate(advocate_list):
                with cols[i % 2]:
                    st.markdown(
                        f"""
                        <div style='padding: 15px; border: 1px solid #e0e0e0; border-radius: 5px; margin-bottom: 10px;'>
                            <h4 style='margin: 0; color: #1a237e;'>{advocate['name']}</h4>
                            <p style='margin: 5px 0;'><strong>Specialization:</strong> {advocate['specialization']}</p>
                            <p style='margin: 5px 0;'><strong>Experience:</strong> {advocate['experience']}</p>
                            <p style='margin: 5px 0;'><strong>Contact:</strong> {advocate['contact']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info(f"No advocates found in {user_location}. Please try a different location.")

def main():
    init_session_state()
    show_sidebar()
    
    if st.session_state.current_page == 'login':
        show_login_page()
    elif st.session_state.current_page == 'register':
        show_register_page()
    elif st.session_state.current_page == 'profile':
        if st.session_state.user:
            show_profile_page()
        else:
            st.session_state.current_page = 'login'
            st.rerun()
    elif st.session_state.current_page == 'main':
        if not st.session_state.user:
            st.session_state.current_page = 'login'
            st.rerun()
        else:
            show_main_page()
    elif st.session_state.current_page == 'interactive_qa':
        if not st.session_state.user:
            st.session_state.current_page = 'login'
            st.rerun()
        else:
            show_interactive_qa()
    elif st.session_state.current_page == 'nearby_courts':
        if not st.session_state.user:
            st.session_state.current_page = 'login'
            st.rerun()
        else:
            show_nearby_courts()
    elif st.session_state.current_page == 'nearby_advocates':
        if not st.session_state.user:
            st.session_state.current_page = 'login'
            st.rerun()
        else:
            show_nearby_advocates()

if __name__ == "__main__":
    main()
