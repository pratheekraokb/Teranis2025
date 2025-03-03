import pandas as pd
import os, io, re, json
import random, string
import streamlit as st
JSON_FOLDER = "JSON_Files"
os.makedirs(JSON_FOLDER, exist_ok=True)

class ExcelReferralFunc:
    def clean_column_name(column_name):
        return column_name.strip().lower()
    
    def get_column_values(file_path, sheet_name=None, target_column="Referral"):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    
        if isinstance(df, dict):
            first_sheet_name = list(df.keys())[0]  # Get the first sheet's name
            df = df[first_sheet_name]  # Select the first sheet
            
        df.columns = [ExcelReferralFunc.clean_column_name(col) for col in df.columns]
    
        target_column_cleaned = ExcelReferralFunc.clean_column_name(target_column)
    
        if target_column_cleaned in df.columns:
            return df[target_column_cleaned].dropna().tolist()
        else:
            return []
    


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

    def get_referral_details(ref_list, filename):
        """Processes referral codes and returns structured details sorted by total referrals (descending order)."""
        referral_details = {}
    
        for each_ref_code in ref_list:
            decrypted_data = ExcelReferralFunc.decryptCode(each_ref_code)
    
            if "error" in decrypted_data:
                continue  # Skip invalid codes
    
            if each_ref_code not in referral_details:
                referral_details[each_ref_code] = {
                    "total_times_referred": 1,
                    "referral_details": decrypted_data
                }
            else:
                referral_details[each_ref_code]["total_times_referred"] += 1
    
        # Sorting by 'total_times_referred' in descending order
        sorted_referral_details = dict(sorted(referral_details.items(), key=lambda x: x[1]["total_times_referred"], reverse=True))
        with open(f"Results/{filename}.json", "w", encoding="utf-8") as json_file:
            json.dump(sorted_referral_details, json_file, indent=4)
        return sorted_referral_details
    
    def save_uploaded_json(uploaded_files):
        file_paths = []
        

        for file in uploaded_files:
            file_path = os.path.join(JSON_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        return file_paths

    def merge_json_files(file_paths):
        merged_data = {}

        for file_path in file_paths:
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)  # ✅ Now correctly opening and reading the file

                for key, value in data.items():
                    if key not in merged_data:
                        merged_data[key] = value  # Copy full structure
                    else:
                        # Sum total_times_referred
                        merged_data[key]["total_times_referred"] += value["total_times_referred"]

            except Exception as e:
                st.error(f"❌ Error reading file {file_path}: {e}")  # ✅ Now correctly referencing file path

        return merged_data


class Certification:
    # Function to generate a unique 10-character code
    def generate_unique_code(prefix, existing_codes):
        """Generate a unique 10-character code with the given prefix."""
        while True:
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
            unique_code = prefix + random_part
            if unique_code not in existing_codes:
                existing_codes.add(unique_code)
                return unique_code


    # Function to process the Excel file
    def process_excel(file, prefix):
        """Process Excel file and generate unique codes."""
        df = pd.read_excel(file)

        # Identify column names dynamically
        headers = df.columns.tolist()
        name_col = next((col for col in headers if "name" in col.lower()), None)
        email_col = next((col for col in headers if "email" in col.lower()), None)
        phone_col = next((col for col in headers if "phone" in col.lower()), None)
        dept_col = next((col for col in headers if "department" in col.lower()), None)
        sem_col = next((col for col in headers if "semester" in col.lower()), None)  # Optional

        if not all([name_col, email_col, phone_col, dept_col]):
            st.error("Required columns (Name, Email, Phone, Department) not found!")
            return None, None

        # Set to store unique codes
        existing_codes = set()

        # Generate unique codes
        df["Unique Code"] = df.apply(lambda row: Certification.generate_unique_code(prefix, existing_codes), axis=1)

        # Create JSON data
        json_data = {}
        for _, row in df.iterrows():
            user_data = {
                "name": row[name_col],
                "email": row[email_col],
                "phone": row[phone_col],
                "department": row[dept_col]
            }
            if sem_col:  # Include semester only if it exists
                user_data["semester"] = row[sem_col]

            json_data[row["Unique Code"]] = user_data

        return df, json_data

# Page Configuration
st.set_page_config(page_title="Teranis 2025 - Admin Panel", layout="wide")

# Sidebar - Navigation Menu
st.sidebar.title("📌 Navigation")
menu_options = ["🏠 Home", "📂 Generate Each Event JSON", "📑 Merge JSON", "Unique Code - Certification"]
selected_option = st.sidebar.radio("Select an Option", menu_options)

# Right Sidebar - Branding
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 **Admin Panel**")
    st.markdown("#### 🏆 **Teranis 2025 - Technical Fest**")
    st.markdown("##### 📍 LBS College of Engineering, Kasaragod")
    st.markdown("---")

# Main Content - Dynamic Based on Selection
if selected_option == "🏠 Home":
    st.title("🏆 **Teranis 2025 - National Level Technical Fest**")
    st.markdown("""
        ### 📍 **LBS College of Engineering, Kasaragod**  
        **Department of Computer Science & Engineering**  
        
        🎉 **Welcome to the Admin Panel!**  
        Manage event referral data, generate JSON files, and handle event analytics seamlessly.
    """)

elif selected_option == "📂 Generate Each Event JSON":
    st.title("📊 **Referral JSON Generator**")
    st.write("Upload an Excel file to generate referral stats for events.")

    # File Upload Section
    uploaded_file = st.file_uploader("📂 **Upload Excel File** (Only .xlsx)", type=["xlsx"], accept_multiple_files=False)

    # File Preview & Remove Option
    if uploaded_file:
        st.success("✅ File Uploaded Successfully!")
        df = pd.read_excel(uploaded_file)  # Read Excel File

        # Display Excel Preview
        st.subheader("📌 **Excel Preview**")
        st.dataframe(df.head(10))  # Show top 10 rows

        # Remove File Button
        if st.button("❌ Remove File"):
            uploaded_file = None
            st.warning("File Removed! Upload a new one.")

    # Event Name Input
    event_name = st.text_input("🏅 **Enter Event Name**", placeholder="Example: Coding Challenge")

    # Generate Referral JSON Button
    if st.button("🚀 Generate Referral JSON Stats"):
        if uploaded_file and event_name.strip():
            # Save Excel File
            save_path = f"Events_Excel/{event_name}.xlsx"
            os.makedirs("Events_Excel", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process Referral Data
            ref_list = ExcelReferralFunc.get_column_values(save_path)
            result = ExcelReferralFunc.get_referral_details(ref_list, event_name)

            st.success(f"🎉 Success! Referral JSON generated for **{event_name}**.")
        else:
            st.error("⚠️ Please upload an Excel file and enter an Event Name.")

elif selected_option == "📑 Merge JSON":
    st.title("📑 **Merge Referral JSON of Various Events**")
    st.write("Upload multiple JSON files to combine into a single structured dataset.")

    uploaded_json_files = st.file_uploader(
        "📂 **Upload Multiple JSON Files**", type=["json"], accept_multiple_files=True
    )

    if uploaded_json_files:
        st.subheader("📌 **Uploaded Files (Saved in `JSON_Files/`)**")
        saved_paths = ExcelReferralFunc.save_uploaded_json(uploaded_json_files)

        for file in uploaded_json_files:
            st.write(f"📄 {file.name}")

        if st.button("🔄 Merge JSON Files"):
            merged_data = ExcelReferralFunc.merge_json_files(saved_paths)

            # Display merged JSON
            st.subheader("✅ **Merged JSON Output**")
            st.json(merged_data)

            # Save merged JSON to file
            JSON_FOLDER = "Results"
            merged_file_path = os.path.join(JSON_FOLDER, "merged_referral_data.json")
            with open(merged_file_path, "w") as f:
                json.dump(merged_data, f, indent=4)

            st.success(f"🎉 Merged JSON successfully saved in `Results/merged_referral_data.json`.")
            st.download_button(
                "⬇️ Download Merged JSON", 
                data=json.dumps(merged_data, indent=4),
                file_name="merged_referral_data.json",
                mime="application/json"
            )

elif selected_option == "Unique Code - Certification":
    st.subheader("Options")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    prefix = st.text_input("Enter 3-Letter Prefix", max_chars=3, value="ABC").upper()
    output_excel_name = st.text_input("Excel Filename", value="output.xlsx")
    output_json_name = st.text_input("JSON Filename", value="output.json")

    if st.button("Generate Unique Codes"):
        if uploaded_file is None:
            st.warning("Please upload an Excel file first.")
        elif len(prefix) != 3:
            st.warning("Prefix must be exactly 3 letters.")
        else:
            # Process the file
            df, json_data = Certification.process_excel(uploaded_file, prefix)

            if df is not None:
                # Ensure directories exist
                os.makedirs("Certifications/Excel", exist_ok=True)
                os.makedirs("Certifications/JSON", exist_ok=True)

                # Define file paths
                excel_path = f"Certifications/Excel/{output_excel_name}"
                json_path = f"Certifications/JSON/{output_json_name}"

                # Save Excel file
                df.to_excel(excel_path, index=False, engine="openpyxl")

                # Save JSON file
                with open(json_path, "w") as json_file:
                    json.dump(json_data, json_file, indent=4)

                # Display success message
                st.success("✅ Unique codes generated successfully!")
                st.write(f"📂 Excel File Saved at: `{excel_path}`")
                st.write(f"📂 JSON File Saved at: `{json_path}`")

                # Show Data Preview
                st.subheader("📌 Data Preview")
                st.dataframe(df.head())
# Footer
st.markdown("---")
st.markdown("© **Teranis 2025 - LBS College of Engineering**")