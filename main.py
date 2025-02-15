import streamlit as st
import re
st.set_page_config(page_title="Teranis Fest 2025", page_icon="🚀")
class TeranisFunc:
    def decryptCode(code):
        try:
            pattern = r"^([1-8])([A-C]?)([A-Z]+)(\d{2})$"
            match = re.match(pattern, code)
            
            if not match:
                raise ValueError("Invalid code format")
            
            semester = f"S{match.group(1)}"
            class_section = match.group(2) if match.group(2) else ""
            department = match.group(3)
            roll_number = match.group(4)
            
            valid_departments = {"CS", "EC", "EEE", "ME", "CE", "IT"}  # Add more as needed
            if department not in valid_departments:
                raise ValueError("Invalid department")
            
            return {
                "semester": semester,
                "class_section": class_section,
                "department": department,
                "roll_number": roll_number
            }
        except Exception as e:
            return {"error": str(e)}

    def encryptCode(semester=8, department="CS", class_section="", roll_num=1):
        try:
            semester = int(semester)
            roll_num = int(roll_num)
            
            if semester < 1 or semester > 8:
                raise ValueError("Semester must be between 1 and 8")
            
            valid_departments = {"CS", "EC", "EEE", "ME", "CE", "IT"}
            if department not in valid_departments:
                raise ValueError("Invalid department")
            
            if class_section and class_section not in {"A", "B", "C"}:
                raise ValueError("Invalid class section")
            
            if roll_num < 1 or roll_num > 99:
                raise ValueError("Roll number must be between 1 and 99")
            
            return f"{semester}{class_section}{department}{roll_num:02d}"
        except Exception as e:
            return {"error": str(e)}
# Custom CSS for Stylish UI
st.markdown(
    """
    <style>
    /* Gradient Background */
    .main {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        color: white;
    }
    *{
    padding: 0px;
    margin: 0px;
    }
    
    /* Title Styling */
    h1 {
        text-align: center;
        color: #FFD700;
        font-size: 3.8em;
        font-weight: bold;
        font-family: 'Poppins', sans-serif;
        text-shadow: 3px 3px 10px rgba(255, 215, 0, 0.7);
    }

    /* Subheading Styling */
    h2 {
        text-align: center;
        color: #00FFFF;
        font-size: 1.7em;
        font-style: italic;
        font-family: 'Roboto', sans-serif;
        text-shadow: 2px 2px 6px rgba(0, 255, 255, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title
st.markdown("<h1>🚀 TERANIS 2025</h1>", unsafe_allow_html=True)

# Subheading
st.markdown("<h2>🔥 LBS College of Engineering, Kasaragod - National Level Technical Fest 🔥</h2>", unsafe_allow_html=True)

# Section
st.markdown("### 🎟 Referral Code Generation")

# Dropdowns with Icons
semester = st.selectbox("📚 Select Your Semester", [f"S{i}" for i in range(1, 9)], index=0)
dept = st.selectbox("🏛 Select Your Department", ["CS", "EC", "EEE", "IT", "ME", "CE"])
division = st.selectbox("📌 Select Your Division", ["None", "A", "B", "C", "D"])
roll_number = st.number_input("🔢 Enter Your Roll Number", min_value=1, step=1)

# Convert Semester (S1 -> 1, S2 -> 2, etc.)
semester_value = int(semester[1:])  

# Convert "None" to an empty string
division_value = "" if division == "None" else division

# Submit Button
if st.button("🚀 Generate Referral Code 🎯"):
    referral_code = TeranisFunc.encryptCode(semester_value, dept, division_value, roll_number)
    st.success(f"🎉 Your Referral Code is: **{referral_code}**")