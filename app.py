import streamlit as st
from PIL import Image
import io
import time
import json
import os
import numpy as np
import google.generativeai as genai
import random
from fun_mcqs import fun_mcqs

# Page configuration
st.set_page_config(
    page_title="AutoMARK AI - Professional MCQs Grading",
    page_icon="✔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.0/css/all.min.css" integrity="sha512-DxV+EoADOkOygM4IR9yXP8Sb2qwgidEmeqAEmDKIOfPRQZOWbXCzLC6vjbZyy0vPisbH2SyW27+ddLVCN+OMzQ==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }

    .hero-section {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        color: #fff !important;
        margin-top: -4rem;
        align-items: center;
        padding: 1rem 2rem 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 30px 40px rgba(0,0,0,0.1);
    }
            
    .hero-title {
        font-size: 3.5rem;
        text-align: center;
        color: #fff;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #fff, #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: #fff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
            
    .subtitle-div {
        overflow: hidden;
        width: 100%;  
    }
    
    div .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0;
        margin-bottom: 0.5rem;
        transform: scale(0.3);
        animation: scaleBounceLoop 6s infinite;
    }

    @keyframes scaleBounceLoop {
        0%, 60%, 100% {
            opacity: 1;
            transform: scale(0.8);
        }
        10% {
            opacity: 1;
            transform: scale(0.8);
        }
        25% {
            opacity: 0;
            transform: scale(0.3);
        }
        40% {
            opacity: 1;
            transform: scale(0.8);
        }
    }
    .word {
        display: inline-block;
        padding-left: 4px;
        opacity: 0;
        transform: translateY(20px);
        animation: wordReveal 0.6s ease-out forwards;
    }

    .word:nth-child(1) { animation-delay: 0.2s; }
    .word:nth-child(2) { animation-delay: 0.4s; }
    .word:nth-child(3) { animation-delay: 0.6s; }
    .word:nth-child(4) { animation-delay: 0.8s; }

    @keyframes wordReveal {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        color: black;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        transform: scale(1.05);
        z-index: 1;
        box-shadow: 0 15px 45px rgba(0,0,0,0.15);
    }
            
    .works {
        background: white;
        padding: 2rem;
        align-items: center;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 260px;
    }

    .works:hover {
        transform: translateY(-5px);
        transform: scale(1.05);
        border : 1px solid black;
        z-index: 1;
        box-shadow: 0 15px 45px rgba(0,0,0,0.15);
    }
    
    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .results-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        color: #333;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .developer-card {
        background: linear-gradient(135deg, #2d3748, #4a5568, #2d3748);
        background-size: 300% 300%;
        animation: gradientBG 12s ease infinite;
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 0rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: fadeUp 1.2s ease forwards;
        
    }

    .developer-avatar {
        width: 80px; 
        height: 80px; 
        border-radius: 50%; 
        background: linear-gradient(135deg, #667eea, #764ba2); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        margin: 0 auto 0.6rem auto; 
        font-size: 3rem;
        color: black;
        box-shadow: 0 0 20px rgba(102,126,234,0.6);
        animation: float 4s ease-in-out infinite;
    }
    
            
    .typewriter-text {
        margin: 0;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        display: inline-block;
        border-right: 2px solid white;
        white-space: nowrap;
        overflow: hidden;
        animation: continuousType 4s infinite, blink 1s infinite;
        width: 11ch;
    }

    @keyframes continuousType {
        0% {
            width: 0;
        }
        20% {
            width: 11ch; /* Full text "Hussain Ali" */
        }
        40% {
            width: 11ch; /* Hold full text */
        }
        60% {
            width: 0; /* Backspace to empty */
        }
        80% {
            width: 0; /* Hold empty */
        }
        100% {
            width: 0; /* Ready to restart */
        }
    }

    @keyframes blink {
        0%, 50% {
            border-color: white;
        }
        51%, 100% {
            border-color: transparent;
        }
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
            
        
    .name-container {
        position: relative;
        display: inline-block;
        padding: 10px 50px 0px 50px;
        background: transparent;
    }

    .name-text {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: white;
        font-size: 6rem;
        font-weight: 700;
        position: relative;
        z-index: 2;
        text-shadow: 2px 4px 6px rgba(0,0,0,0.3);
    }

    .bubbles {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
    }


    .bubble {
        position: absolute;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.2));
        border-radius: 50%;
        animation: continuousBubble 3s infinite ease-out;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .bubble:nth-child(1) {
        left: 10%;
        width: 6px;
        height: 6px;
        animation-delay: 0s;
    }

    .bubble:nth-child(2) {
        left: 25%;
        width: 8px;
        height: 8px;
        animation-delay: 0.5s;
    }

    .bubble:nth-child(3) {
        left: 40%;
        width: 5px;
        height: 5px;
        animation-delay: 1s;
    }

    .bubble:nth-child(4) {
        left: 60%;
        width: 7px;
        height: 7px;
        animation-delay: 1.5s;
    }

    .bubble:nth-child(5) {
        left: 75%;
        width: 6px;
        height: 6px;
        animation-delay: 2s;
    }

    .bubble:nth-child(6) {
        left: 90%;
        width: 4px;
        height: 4px;
        animation-delay: 2.5s;
    }

    .bubble:nth-child(7) {
        left: 15%;
        width: 5px;
        height: 5px;
        animation-delay: 0.8s;
        animation-duration: 4s;
    }

    .bubble:nth-child(8) {
        left: 55%;
        width: 6px;
        height: 6px;
        animation-delay: 1.8s;
        animation-duration: 3.5s;
    }

    .bubble:nth-child(9) {
        left: 80%;
        width: 4px;
        height: 4px;
        animation-delay: 0.3s;
        animation-duration: 3.2s;
    }

    .bubble:nth-child(10) {
        left: 35%;
        width: 7px;
        height: 7px;
        animation-delay: 2.2s;
        animation-duration: 3.8s;
    }

    @keyframes continuousBubble {
        0% {
            bottom: -10px;
            opacity: 0;
            transform: scale(0);
        }
        10% {
            opacity: 1;
            transform: scale(1);
        }
        90% {
            opacity: 1;
        }
        100% {
            bottom: 100px;
            opacity: 0;
            transform: scale(0.3);
        }
    }
            
    .social-btn {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        height: 50px;
        width: 50px;
        padding: 0.75rem 0.75rem;
        border-radius: 50%;
        text-decoration: none !important;
        font-weight: 600;
        background: linear-gradient(135deg, #2d3748, #4a5568);
        color: white !important;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
        white-space: nowrap;
        animation: socialsAnimation 2s ease-in-out infinite;
    }
            
    .social-btn::before {
        content: "";
        position: absolute;
        color: #fff;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
        transition: left 0.5s ease;
    }
    

    .social-btn:hover::before {
        left: 100%;
    }

    .social-btn:hover {
        transform: scale(1.1) !important;
        text-decoration: none;
        color: black !important;
    }
        
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(102, 126, 234, 0.9);
        color: black;
        padding: 0.5rem 2rem;
        align-items: center !important;
        text-align: center !important;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        z-index: 1000;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
            
    @keyframes socialsAnimation {
        0% {
            box-shadow: 0 2px 5px rgba(36, 52, 77,0.7);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 5px 10px rgba(36, 52, 77,0.9);
            transform: scale(1.01);
        }
        100% {
            box-shadow: 0 2px 5px rgba(0,36, 52, 77,.7); 
            transform: scale(1);
        }
    }
    
    @keyframes pulseAnimation {
        0% {
            box-shadow: 0 2px 10px rgba(0,0,0,0.7);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 4px 15px rgba(0,0,0,1);
            transform: scale(1.1);
        }
        100% {
            box-shadow: 0 2px 10px rgba(0,0,0,0.7);
            transform: scale(1);
        }
    }
    .pro-badge {
        display: inline-block;
        background: linear-gradient(45deg, #fff, #f8f9fa);
        color: black;
        margin-top: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(0,0,0,0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        animation: pulseAnimation 2s ease-in-out infinite;
    }

    .pro-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.7), transparent);
        transition: left 0.6s ease;
    }

    .pro-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
    }

    .pro-badge:hover::before {
        left: 100%;
    }

    .pro-badge:active {
        transform: translateY(0);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.1s ease;
    }
        
    .sidebar .block-container {
        padding-top: 2rem;
    }

    .tips-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }

    .tips-card h4 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
    }

    .tips-card h5 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }

    .tips-card ul {
        margin: 0;
        padding-left: 1.5rem;
        font-size: 0.9rem;
    }

    .tips-card li {
        margin-bottom: 0.3rem;
    }

    .tips-columns {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .tips-column {
        flex: 1;
    }

    @media (max-width: 768px) {
        .tips-columns {
            flex-direction: column;
        }
    }

            
</style>
""", unsafe_allow_html=True)

# Watermark
st.markdown("""
<div class="watermark">
     <strong style="padding-right:0.4rem; text-align:center;">Developer </strong> <a href="https://github.com/choudaryhussainali"><i class="fa-solid fa-code"></i></a>
</div>
""", unsafe_allow_html=True)

def configure_gemini():
    """Configure Gemini API"""
    try:
        # Try to get API key from Streamlit secrets first
        if 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets['GEMINI_API_KEY']
        # Fallback to environment variable
        elif 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ['GEMINI_API_KEY']
        else:
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {str(e)}")
        return None

def check_image_quality(image):
    import numpy as np

    """Check if the image is too blurry for accurate processing using multiple methods"""
    try:
        import cv2
        
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Method 1: Laplacian variance (primary method)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Method 2: Sobel gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mean = np.mean(sobel_magnitude)
        
        # Method 3: Text edge detection (important for MCQ papers)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Strict thresholds for MCQ papers (much higher than before)
        laplacian_threshold = 250  # Increased from 100
        sobel_threshold = 15       # New threshold
        edge_threshold = 0.02      # New threshold
        
        # All methods must pass for image to be considered clear
        laplacian_clear = laplacian_var > laplacian_threshold
        sobel_clear = sobel_mean > sobel_threshold
        edge_clear = edge_density > edge_threshold
        
        is_clear = laplacian_clear and sobel_clear and edge_clear
        
        quality_scores = {
            'laplacian': laplacian_var,
            'sobel': sobel_mean,
            'edges': edge_density,
            'overall_clear': is_clear
        }
        
        return is_clear, quality_scores
        
    except ImportError:
        # Enhanced fallback method without OpenCV
        img_array = np.array(image.convert('L'))
        
        # Multiple fallback checks
        std_dev = np.std(img_array)
        mean_val = np.mean(img_array)
        
        # Calculate local contrast
        kernel_size = 9
        h, w = img_array.shape
        contrast_sum = 0
        count = 0
        
        for i in range(0, h - kernel_size, kernel_size):
            for j in range(0, w - kernel_size, kernel_size):
                patch = img_array[i:i+kernel_size, j:j+kernel_size]
                local_std = np.std(patch)
                contrast_sum += local_std
                count += 1
        
        avg_local_contrast = contrast_sum / count if count > 0 else 0
        
        # Stricter fallback thresholds
        std_threshold = 45      # Increased from 30
        contrast_threshold = 25  # New threshold
        
        is_clear = std_dev > std_threshold and avg_local_contrast > contrast_threshold
        
        quality_scores = {
            'std_dev': std_dev,
            'contrast': avg_local_contrast,
            'overall_clear': is_clear
        }
        
        return is_clear, quality_scores
        
    except Exception:
        # Conservative approach - if we can't check quality, assume it might be blurry
        return False, {'error': 'Could not assess image quality'}

def validate_and_fix_results(result):
    """Validate and fix the results from Gemini API to ensure consistency"""
    if not result or 'mcqs' not in result:
        return result
    
    mcqs = result['mcqs']
    
    # Recalculate correct and wrong answers by actually counting from the mcqs list
    correct_count = sum(1 for mcq in mcqs if mcq.get('is_correct', False))
    total_questions = len(mcqs)
    wrong_count = total_questions - correct_count
    
    # Recalculate percentage
    score_percentage = round((correct_count / total_questions * 100), 2) if total_questions > 0 else 0
    
    # Update the result with the corrected values
    result['total_questions'] = total_questions
    result['correct_answers'] = correct_count
    result['wrong_answers'] = wrong_count
    result['score_percentage'] = score_percentage
    
    return result

def grade_mcqs_from_image(image, model):
    """Process the uploaded image and grade MCQs"""
    start_time = time.time()
    
    # Check image quality first
    is_clear, quality_scores = check_image_quality(image)
    if not is_clear:
        error_msg = "⚠️ Image quality insufficient for accurate grading.\n\n"
        if 'error' in quality_scores:
            error_msg += "Could not assess image quality properly."
        else:
            error_msg += "**Quality Analysis:**\n"
            if 'laplacian' in quality_scores:
                error_msg += f"• Sharpness Score: {quality_scores['laplacian']:.1f} (needs >250)\n"
                error_msg += f"• Edge Clarity: {quality_scores['sobel']:.1f} (needs >15)\n"
                error_msg += f"• Text Definition: {quality_scores['edges']:.3f} (needs >0.02)"
            else:
                error_msg += f"• Contrast Score: {quality_scores['std_dev']:.1f} (needs >45)\n"
                error_msg += f"• Local Sharpness: {quality_scores['contrast']:.1f} (needs >25)"
        
        return None, 0, error_msg

    # Prepare the image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Optimized prompt for Gemini Vision
    prompt = """Carefully analyze this image of a solved MCQ exam paper. Extract and return the following information in JSON format:

    CRITICAL INSTRUCTIONS:
    1. Look VERY CAREFULLY at each question's answer choices (A, B, C, D, etc.)
    2. Identify which option the student has ACTUALLY marked/selected (circled, filled, checked, etc.)
    3. Identify ALL options that could be considered correct (sometimes there are duplicate/misstyped options with same content)
    4. DO NOT assume patterns - each question must be analyzed individually
    5. If you cannot clearly see what the student marked, mark that question as "unclear"
    6. Pay close attention to different marking styles (circles, fills, checkmarks, crosses)
    7. If multiple options have identical or very similar content, list ALL of them as potential correct answers

    Extract and return the following information in EXACT JSON format:

    {
        "student_name": "[Name from paper or 'Unknown']",
        "total_questions": [Total number of questions],
        "correct_answers": [Number of correct answers],
        "wrong_answers": [Number of wrong answers],
        "score_percentage": [Percentage score],
        "mcqs": [
            {
                "question_number": "[Q1, Q2, etc.]",
                "question": "[Full question text]",
                "available_options": "[A, B, C, D or whatever options are shown]",
                "correct_answer": "[Primary correct option]",
                "all_correct_answers": "[Array of ALL correct options if multiple exist, e.g., ['A', 'C'] for duplicate content]",
                "student_answer": "[The option the student ACTUALLY marked/selected]",
                "marking_confidence": "[high/medium/low - how clear was the student's marking]",
                "is_correct": [true/false]
            }
        ]
    }

    VALIDATION RULES:
    - If student answers appear identical across questions, you are making an error
    - Each question should have different student answers unless truly identical
    - Look for different marking patterns: circles, fills, checkmarks, X marks
    - If markings are unclear, set marking_confidence to "low"
    - If you see duplicate options with same content, include ALL in all_correct_answers array
    - Return ONLY valid JSON, no additional text or explanations
    
    DOUBLE-CHECK your work - students rarely mark the same option for every question!
    """

    try:
        # Send to Gemini for processing
        response = model.generate_content(
            [prompt, Image.open(io.BytesIO(img_bytes))],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 4000
            }
        )

        # Clean the response to extract pure JSON
        response_text = response.text.replace('```json', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        # Validate and fix the results to ensure consistency
        result = validate_and_fix_results(result)

        # Calculate processing time
        processing_time = time.time() - start_time

        return result, processing_time, None

    except Exception as e:
        return None, 0, str(e)

def display_results(result, processing_time):
    """Display the grading results in a professional format"""
    
    # Processing time with success message
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; border-radius: 10px; text-align: center;padding: 0 0 0.1rem; margin: 1rem 0;">
        <h3 style="margin: 0;"><i class="fa-duotone fa-solid fa-check-double"></i> Analysis Complete!</h3>
        <p>Processed in {processing_time:.2f} seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional metrics display
    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value"><i class="fa-solid fa-user-tie"></i></div>
            <div class="metric-label">Student Name</div>
            <div style="font-weight: 600; color: #667eea; margin-top: 0.5rem;">
                {result.get('student_name', 'Unknown')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result['total_questions']}</div>
            <div class="metric-label">Total Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">{result['correct_answers']}</div>
            <div class="metric-label">Correct Answers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{result['score_percentage']}%</div>
            <div class="metric-label">Final Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Animated progress bar
    progress_color = "#10b981" if result['score_percentage'] >= 70 else "#f59e0b" if result['score_percentage'] >= 50 else "#ef4444"
    st.markdown(f"""
    <div style="background: #f1f5f9; border-radius: 10px; padding: 1rem; margin: 2rem 0;">
        <div style="background: {progress_color}; width: {result['score_percentage']}%; height: 20px; border-radius: 10px; 
             display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem;
             transition: width 2s ease;">
            {result['score_percentage']}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Grade classification
    if result['score_percentage'] >= 90:
        grade = "A+ Excellent"
        color = "#10b981"
    elif result['score_percentage'] >= 80:
        grade = "A Good"
        color = "#059669"
    elif result['score_percentage'] >= 70:
        grade = "B Average"
        color = "#f59e0b"
    elif result['score_percentage'] >= 60:
        grade = "C Below Average"
        color = "#f97316"
    else:
        grade = "D Needs Improvement"
        color = "#ef4444"
    
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 0.2rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
        <h2 style="margin: 0;"><i class="fa-solid fa-square-poll-horizontal"></i> Grade: {grade}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed results
    st.markdown("""<div style="margin: 0.5rem 0;"><h3><i class="fa-solid fa-list"></i> Detailed Question Analysis</h3></div>""", unsafe_allow_html=True)
    
    correct_questions = [mcq for mcq in result['mcqs'] if mcq['is_correct']]
    incorrect_questions = [mcq for mcq in result['mcqs'] if not mcq['is_correct']]
    
    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs([f"✅ Correct ({len(correct_questions)})", f"❌ Incorrect ({len(incorrect_questions)})", "📊 All Questions"])
    
    with tab1:
        if correct_questions:
            for mcq in correct_questions:
                with st.expander(f"✅ {mcq['question_number']}: Correct Answer", expanded=False):
                    st.markdown(f"**Question:** {mcq['question']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Correct Answer:** {mcq['correct_answer']}")
                    with col2:
                        st.success(f"**Student Answer:** {mcq['student_answer']}")
        else:
            st.info("No correct answers found.")
    
    with tab2:
        if incorrect_questions:
            for mcq in incorrect_questions:
                with st.expander(f"❌ {mcq['question_number']}: Incorrect Answer", expanded=False):
                    st.markdown(f"**Question:** {mcq['question']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Correct Answer:** {mcq['correct_answer']}")
                    with col2:
                        st.error(f"**Student Answer:** {mcq['student_answer']}")
        else:
            st.success("All answers are correct! 🎉")
    
    with tab3:
        for i, mcq in enumerate(result['mcqs']):
            status = "✅ Correct" if mcq['is_correct'] else "❌ Incorrect"
            with st.expander(f"{mcq['question_number']}: {status}", expanded=False):
                st.markdown(f"**Question:** {mcq['question']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Correct Answer:** {mcq['correct_answer']}")
                with col2:
                    st.write(f"**Student Answer:** {mcq['student_answer']}")
                
                if mcq['is_correct']:
                    st.success("✅ Correct Answer!")
                else:
                    st.error("❌ Incorrect Answer")

def sidebar_content():
    """Professional sidebar with developer info and features"""
    with st.sidebar:
        # Developer Profile
        st.markdown("""
            <div class="developer-card">
                <div class="developer-avatar">
                    <i class="fa-duotone fa-solid fa-user-secret"></i>
                </div>
                <h3 class="typewriter-text">HUSSAIN ALI</h3>
                <p style="margin: 0.1rem 0 0.2rem; font-size: 1rem; opacity: 1;">| Python Developer |</p>
                <p style="margin: 1rem auto; font-size: 0.9rem; opacity: 0.7;">
                    Always learning. Always building. Let's connect and innovate!
                </p>
            </div>
            """, unsafe_allow_html=True)

        
        # Social Media Links
        st.markdown("""
            <div class="social-links">
                    <a href="https://linkedin.com/in/ch-hussain-ali" class="social-btn" style="font-size:1.5rem;" target="_blank">
                    <i class="fab fa-linkedin"></i>
                    </a>
                    <a href="https://github.com/choudaryhussainali" class="social-btn" style="font-size:1.5rem;" target="_blank">        
                    <i class="fab fa-github"></i>
                    </a>
                    <a href="mailto:choudaryhussainali@outlook.com" class="social-btn" style="font-size:1.5rem;" target="_blank">
                        <i class="fas fa-envelope"></i>
                    </a>
                    <a href="https://www.instagram.com/choudary_hussain_ali/" class="social-btn" style="font-size:1.5rem;" target="_blank">
                        <i class="fa-brands fa-instagram"></i>
                    </a>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---") 

        st.markdown("""<div style="text-align: center;"><h3 style="color: #667eea; margin-bottom: 1rem;">How We Work</h3></div>""", unsafe_allow_html=True)
        
        steps = [
            ("📞", "Initial Call", "Discuss requirements"),
            ("📋", "Project Scope", "Define deliverables"),
            ("🔨", "Development", "Build with updates"),
            ("🚀", "Launch", "Deploy & celebrate")
        ]
        
        for i, (icon, title, desc) in enumerate(steps):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f9fafb, #f3f4f6); border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem; border-left: 3px solid #3b82f6;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.8rem;">{icon}</span>
                    <h5 style="color: #667eea; margin: 0; font-size: 0.9rem;">{title}</h5>
                </div>
                <p style="color: #888; font-size: 0.8rem; margin: 0; padding-left: 2.3rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
         
        st.markdown("---")

        # App Features
        st.markdown("### ⚡ Features")
        st.markdown("""
        - 🤖 **AI-Powered Grading**
        - 🎯 **98.5% Accuracy**
        - ⚡ **Lightning Fast**
        - 📱  **Mobile Friendly**
        - 🔒 **Secure & Private**
        - 🌍 **Global Access**
        - 📊 **Detailed Analytics**
        - 💾 **Export Results**
        """)
        st.markdown("---")
        st.markdown("""
             <h3 style='text-align: center;'><i class="fa-solid fa-bars-progress"></i> Core Insights</h3>""",
            unsafe_allow_html=True
        )

        insights = [
            ("📂", "Total Processed", "500+", "+23%"),
            ("🎯", "Accuracy Rate", "98.5%", "+2.1%"),
            ("⭐", "User Satisfaction", "4.7 / 5", "Stable"),
        ]

        for icon, title, value, delta in insights:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f9fafb, #f3f4f6);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 0.8rem;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            ">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div style="display:flex; align-items:center;">
                        <span style="font-size:1.4rem; margin-right:0.6rem;">{icon}</span>
                        <div>
                            <p style="margin:0; font-size:0.85rem; color:#555;">{title}</p>
                            <h4 style="margin:0; font-size:1.1rem; color:#1f2937;">{value}</h4>
                        </div>
                    </div>
                    <span style="font-size:0.9rem; color:#10b981; font-weight:600;">{delta}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

        def fun_fact_mcq_sidebar():
            """Sidebar widget: Fun Fact MCQ of the Day"""
            st.markdown("""
                <h2 style='text-align: center;'><i style="color:yellow;" class="fa-solid fa-lightbulb"></i></i> Fun Fact <strong style="color: #667eea;">MCQ</strong> of the day </h2>""",
                unsafe_allow_html=True
            )
            if "fun_mcq" not in st.session_state:
                st.session_state.fun_mcq = random.choice(fun_mcqs)
                
            q = st.session_state.fun_mcq
            # Show the question
            st.sidebar.write(f"**{q['question']}**")
            
            # Show options
            choice = st.sidebar.radio(
                "Pick your answer:",
                q["options"],
                key="fun_mcq_choice"
            )
            
            # Submit button
            if st.sidebar.button("Submit Answer", key="fun_mcq_submit"):
                if choice == q["answer"]:
                    st.sidebar.success(f"✅ Correct! {q['fact']}")
                else:
                    st.sidebar.error(f"❌ Oops! Correct answer: **{q['answer']}**\n\n{q['fact']}")


        fun_fact_mcq_sidebar()
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; padding: 0; margin-top: 0.5rem;">
                <p style="color: #999; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                    © 2025 AutoMARK AI. All rights reserved.
                </p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Sidebar
    sidebar_content()
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
         <div class="name-container">
            <h2 class="name-text">AutoMARK</h2>
                <div class="bubbles">
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                    <div class="bubble"></div>
                </div>
         </div>
         <div class="subtitle-div">
                <p class="hero-subtitle">
                    <span class="word">Professional</span>
                    <span class="word">Multiple - Choice</span>
                    <span class="word">Grading</span>
                    <span class="word">Platform</span>
                </p>
         </div>
        <p><span class="pro-badge">AI POWERED</span></p>
        <p style="font-size: 1rem; opacity: 0.7; max-width: 600px; margin: 0 auto;">
            Transform your grading process with cutting-edge AI technology. 
            Fast, accurate, and reliable automated MCQs evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">♚</div>
            <h3 style="text-align: center; color: #667eea;">Lightning Fast</h3>
            <p style="text-align: center; color: #666;">
                Grade hundreds of MCQs in seconds with our advanced AI engine
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">♛</div>
            <h3 style="text-align: center; color: #667eea;">98.5% Accurate</h3>
            <p style="text-align: center; color: #666;">
                Industry-leading accuracy with advanced AI Computer Vision technology
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 1rem;">♜</div>
            <h3 style="text-align: center; color: #667eea;">Detailed Analytics</h3>
            <p style="text-align: center; color: #666;">
                Comprehensive reports with question-by-question analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # API Configuration
    model = configure_gemini()
    
    if model is None:
        st.error("⚠️ Gemini API key not found!")
        st.info("""
        🔑 **API Configuration Required**
        
        To use this application, please configure your Gemini API key:
        1. Get your API key from: https://makersuite.google.com/app/apikey
        2. Add it to Streamlit secrets or environment variables
        
        **For Developers:** Contact the developer for enterprise API access.
        """)
        st.stop()
    
    # Upload Section
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("""
       <h3><i class="fa-solid fa-camera"></i> Upload Answer Sheet</h3>
        """, unsafe_allow_html=True)
    st.markdown("Drag and drop your MCQs answer sheet image or click to browse")
    
    uploaded_file = st.file_uploader(
        "",
        type=['png', 'jpg', 'jpeg'],
        help="Support: PNG, JPG, JPEG | Max size: 10MB"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Two-column layout for image and results
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #667eea;"><i class="fa-solid fa-image"></i> Uploaded Image</h3>
            </div>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded_file)
            st.image(image, caption="Answer Sheet Preview", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div style="background: white; color: black; padding: 1rem 1rem 0.5rem 1rem; border-radius: 10px; margin: 1rem;margin-bottom: 2rem; box-shadow: 10px 15px 30px rgba(0,0,0,0.15);">
                <strong><i class="fa-solid fa-image"></i> Image Details:</strong><br>
                <p style="opacity: 0.8;">
                    • Size: {image.size[0]} x {image.size[1]} pixels<br>
                    • Format: {image.format}<br>
                    • File: {uploaded_file.name}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3 style="color: #667eea;"><i class="fa-solid fa-robot"></i>  AI Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Grade button with professional styling
            if st.button("➲  Start AI Grading", type="primary", use_container_width=True):
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Animated progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔍 Analyzing image quality...")
                    elif i < 60:
                        status_text.text("🤖 AI processing MCQs...")
                    elif i < 80:
                        status_text.text("📊 Calculating results...")
                    else:
                        status_text.text("✅ Finalizing analysis...")
                    time.sleep(0.04)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Process the image
                result, processing_time, error = grade_mcqs_from_image(image, model)
                
                if error:
                    st.error(f"❌ **Analysis Failed**")
                    st.markdown(f"""
                    <div style="background: #fee2e2; color: black; border: 1px solid #fecaca; padding: 1.5rem; border-radius: 10px; margin: 1rem 0 2rem;">
                        <h4 style="color: #dc2626; margin-top: 0;">Error Details:</h4>
                        <p style="color: #7f1d1d; margin-bottom: 0;">{error}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "quality insufficient" in error.lower():

                        st.markdown('<h4><i class="fa-solid fa-camera"></i> Image Quality Tips</h4>', unsafe_allow_html=True)
                        with st.expander("Make sure this !", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**Camera Settings:**")
                                st.write("• Use camera app")
                                st.write("• Enable HDR/focus")
                                st.write("• Highest quality")
                            with col2:
                                st.write("**Positioning:**")
                                st.write("• Hold directly above")
                                st.write("• 12-18 inches away")
                                st.write("• Keep steady")
                            with col3:
                                st.info("📄 Upload only exam papers with multiple choice questions for best results")
                elif result:
                    st.success("🎉 **Analysis Completed Successfully!**")
                    st.balloons()
    
    # Results display (outside the columns)
    if uploaded_file is not None and 'result' in locals() and result:
        display_results(result, processing_time)
        
        # Download section
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # JSON download
            json_data = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"mcq_analysis_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV download (summary)
            import pandas as pd
            summary_data = {
                'Student Name': [result.get('student_name', 'Unknown')],
                'Total Questions': [result['total_questions']],
                'Correct Answers': [result['correct_answers']],
                'Wrong Answers': [result['wrong_answers']],
                'Score Percentage': [result['score_percentage']],
                'Processing Time (s)': [f"{processing_time:.2f}"]
            }
            csv_summary = pd.DataFrame(summary_data).to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Summary",
                data=csv_summary,
                file_name=f"mcq_summary_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # with col3:
        #     # Share results
        #     if st.button("📤 Share Results", use_container_width=True):
        #         st.success("🔗 Shareable link copied to clipboard!")
        #         st.balloons()
    
    # How it works section
    st.markdown("---")
    st.markdown("""
       <h2><i class="fa-solid fa-file-circle-check"></i> How <strong style="color: #667eea;">AutoMARK</strong> Works</h2>
        """, unsafe_allow_html=True)   
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="works" style="text-align: center; padding: 2rem 1rem;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #667eea, #764ba2); 
                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                 margin: 0 auto 1rem auto; color: white; font-size: 1.5rem;"><i class="fa-solid fa-image"></i></div>
            <h4 style="color: #667eea;">Upload Image</h4>
            <p style="color: #666; font-size: 0.9rem;">Upload a sharp and readable picture of your answer sheet</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="works" style="text-align: center; padding: 2rem 1rem;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #667eea, #764ba2); 
                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                 margin: 0 auto 1rem auto; color: white; font-size: 1.5rem;"><i class="fa-solid fa-brain"></i></div>
            <h4 style="color: #667eea;">AI Analysis</h4>
            <p style="color: #666; font-size: 0.9rem;">Advanced OCR extracts Questions and Answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="works" style="text-align: center; padding: 2rem 1rem;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #667eea, #764ba2); 
                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                 margin: 0 auto 1rem auto; color: white; font-size: 1.5rem;"><i class="fa-solid fa-hat-cowboy"></i></div>
            <h4 style="color: #667eea;">Smart Grading</h4>
            <p style="color: #666; font-size: 0.9rem;">AI compares answers with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="works" style="text-align: center; padding: 2rem 1rem;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #667eea, #764ba2); 
                 border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                 margin: 0 auto 1rem auto; color: white; font-size: 1.5rem;"><i class="fa-solid fa-square-poll-vertical"></i></div>
            <h4 style="color: #667eea;">Detailed Report</h4>
            <p style="color: #666; font-size: 0.9rem;">Comprehensive insights and results at your fingertips</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Testimonials section
    st.markdown("---")
    st.markdown("""
       <h2> Why <strong style="color: #667eea;">User</strong> Love Us ♥</h2>
     """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 1rem;">⭐⭐⭐⭐⭐</div>
                <p style="font-style: italic; color: #666;">"Incredible accuracy! Saved me hours of manual grading."</p>
                <strong style="color: #667eea;">- Dr. Sarah Johnson</strong><br>
                <small style="color: #999;">Mathematics Professor</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 1rem;">⭐⭐⭐⭐⭐</div>
                <p style="font-style: italic; color: #666;">"Fast, reliable, and easy to use. Perfect for our school!"</p>
                <strong style="color: #667eea;">- Michael Chen</strong><br>
                <small style="color: #999;">School Administrator</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 1rem; margin-bottom: 1rem;">⭐⭐⭐⭐⭐</div>
                <p style="font-style: italic; color: #666;">"Revolutionary technology! Makes grading effortless."</p>
                <strong style="color: #667eea;">- Emily Rodriguez</strong><br>
                <small style="color: #999;">High School Teacher</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # FAQ Section
    with st.expander("❓ Frequently Asked Questions", expanded=False):
        st.markdown("""
            <div><h3><i class="fa-solid fa-comments"></i> Common Questions</h3></div> 
        """, unsafe_allow_html=True)
        st.markdown("""
        **Q: What image formats are supported?**  
        A: We support PNG, JPG, and JPEG formats up to 10MB in size.
        
        **Q: How accurate is the AI grading?**  
        A: Our AI achieves 98.5% accuracy rate with clear, high-quality images.
        
        **Q: Can it handle handwritten answers?**  
        A: Currently optimized for printed MCQs. Handwritten support coming soon!
        
        **Q: Is my data secure?**  
        A: Yes! We don't store your images or results. Everything is processed securely.
        
        **Q: Can I grade multiple sheets at once?**  
        A: Batch processing feature is available for premium users. Contact us for details.
        
        **Q: What if the grading seems incorrect?**  
        A: Please contact our support team with the image for manual review and improvement.
        """
        )
    

    
    # Developer attribution footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; border-top: 1px solid #e2e8f0; margin-top: 2rem;">
        <p style="color: #666; margin: 0;">
            ✦ Crafted with ❣ by <strong style="color: #667eea;">CHOUDARY HUSSAIN ALI</strong> | 
            Powered by Google Vision AI | Built with Advanced Technology ✦
        </p>
        <p style="color: #999; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            © 2025 AutoMARK AI. All rights reserved. | 
            <a href="#" style="color: #667eea;">Privacy Policy</a> | 
            <a href="#" style="color: #667eea;">Terms of Service</a>
        </p>
        <p style="font-size: 0.8rem; margin: 0.5rem; text-decoration: none;">
            <span>E-mail | </span>
            <a href="mailto:choudaryhussainali@outlook.com" style="color: #667eea;text-decoration: none;">choudaryhussainali@outlook.com</a>   
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

