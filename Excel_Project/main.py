import pandas as pd
import os, io, re, json
import random, string
import streamlit as st


# certifications

import os
import glob
import shutil
import zipfile
import pandas as pd
import streamlit as st
import streamlit as st
import cv2
import json
import os
import zipfile
import tempfile
import qrcode
import numpy as np
from difflib import get_close_matches

import cv2
import numpy as np
import qrcode
import json
from PIL import Image



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
            
    def combine_json_files(uploaded_files):
        combined_data = {}

        for uploaded_file in uploaded_files:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, dict):
                    combined_data.update(data)  # Merge JSON objects based on unique keys
                else:
                    st.error(f"Invalid JSON format in file: {uploaded_file.name} (Expected a dictionary)")
            except json.JSONDecodeError:
                st.error(f"Error decoding JSON in file: {uploaded_file.name}")

        return combined_data


    # Function to process the Excel file
    def process_excel(file, prefix, event_name, event_date, certificate_type, notes=""):
        """Process Excel file and generate unique codes."""
        df = pd.read_excel(file)

        # Identify column names dynamically
        headers = df.columns.tolist()
        name_col = next((col for col in headers if "name" in col.lower()), None)
        email_col = next((col for col in headers if "email" in col.lower()), "")
        phone_col = next((col for col in headers if "phone" in col.lower()), "")
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
                "name": str(row.get(name_col, "")).upper(),
                "email": row.get(email_col, ""),
                "phone": row.get(phone_col, ""),
                "department": row.get(dept_col, ""),        # eg CS, EC, 
                "semester": row.get(sem_col, ""),        # Include semester, default to empty string  eg S2 CS A
                "event_name": str(event_name),              
                "event_date": str(event_date),
                "type_of_certificate": str(certificate_type),
                "notes": str(notes)
            }
            if sem_col:  # Include semester only if it exists
                user_data["semester"] = row[sem_col]

            json_data[row["Unique Code"]] = user_data

        return df, json_data
    

    def detect_column_name(df, target_keywords):
        """Finds the best matching column name from a list of possible variations."""
        df_columns = list(df.columns)  # Keep original column names
        lower_to_original = {col.lower().strip(): col for col in df_columns}  # Map lowercase to original
        df_columns_lower = list(lower_to_original.keys())  # Lowercase column names for matching

        for keyword in target_keywords:
            match = get_close_matches(keyword, df_columns_lower, n=1, cutoff=0.7)  # Fuzzy matching
            if match:
                return lower_to_original[match[0]]  # Return original column name
        return None

    def save_uploaded_file(uploaded_file, save_path):
        """Save uploaded file to a specified path."""
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    def extract_zip(uploaded_zip, extract_to="Certificates"):
        """Extracts the uploaded ZIP file to a folder."""
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    def process_certificates(excel_file, certificates_path="Certificates"):
        """Reads the Excel file, checks PNG count, and renames certificate files."""
        try:
            # Read Excel
            df = pd.read_excel(excel_file)
            
            # Detect SL Number column
            sl_number_column = Certification.detect_column_name(df, ["sl number", "sl no", "sl num", "serial no", "sl_num"])
            if not sl_number_column:
                return "Error: Could not find a valid 'SL Number' column."

            unique_code_column = Certification.detect_column_name(df, ["unique code", "code", "unique id"])
            if not unique_code_column:
                return "Error: Could not find a valid 'Unique Code' column."

            # Create dictionary mapping SL Number to Unique Code
            mapping_dict = dict(zip(df[sl_number_column].astype(str), df[unique_code_column].astype(str)))

            if not os.path.isdir(certificates_path):
                return "Error: 'Certificates' folder not found."

            # Get all PNG files
            all_png_files = glob.glob(os.path.join(certificates_path, "**", "*.png"), recursive=True)

            if len(all_png_files) != len(mapping_dict):
                return f"Error: Mismatch - Found {len(all_png_files)} PNGs, expected {len(mapping_dict)}."

            # Rename files
            renamed_files = []
            for sl_number, unique_code in mapping_dict.items():
                old_path = os.path.join(certificates_path, f"{sl_number}.png")
                new_path = os.path.join(certificates_path, f"{unique_code}.png")

                if os.path.isfile(old_path):
                    os.rename(old_path, new_path)
                    renamed_files.append(f"Renamed {old_path} → {new_path}")

            return renamed_files if renamed_files else "No files were renamed."

        except Exception as e:
            return f"Error: {str(e)}"

    def zip_folder(folder_path, output_zip="Processed_Certificates.zip"):
        """Zips the processed folder for download."""
        shutil.make_archive(output_zip.replace(".zip", ""), 'zip', folder_path)
        return output_zip

    def make_logo_transparent(logo_path):
        """
        Loads a logo and removes the white background to make it transparent.
        """
        # logo_cv = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        logo_cv = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        return Image.fromarray(logo_cv)

    def generate_qr_with_logo(qr_url, logo_path):
        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(qr_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill="black", back_color="white").convert("RGBA")

        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            qr_size = qr_img.size[0]
            logo_size = qr_size // 4
            logo = logo.resize((logo_size, logo_size))
            pos = ((qr_size - logo_size) // 2, (qr_size - logo_size) // 2)
            qr_img.paste(logo, pos, mask=logo if logo.mode == "RGBA" else None)
        return qr_img




    def overlay_qr_on_certificate(certificate_path, qr_path, save_path="certificate_with_qr.png"):
        """
        Overlays the generated QR code onto a certificate image based on user selection.
        """
        image = cv2.imread(certificate_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        clone = image.copy()
        rect = []
        drawing = False
        
        def select_rectangle(event, x, y, flags, param):
            nonlocal rect, drawing, clone
            if event == cv2.EVENT_LBUTTONDOWN:
                rect = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                clone = image.copy()
                cv2.rectangle(clone, rect[0], (x, y), (0, 255, 0, 255), 2)
                cv2.imshow("Select QR Area", clone)
            elif event == cv2.EVENT_LBUTTONUP:
                rect.append((x, y))
                drawing = False
                cv2.rectangle(clone, rect[0], rect[1], (0, 255, 0, 255), 2)
                cv2.imshow("Select QR Area", clone)
        
        cv2.namedWindow("Select QR Area", cv2.WINDOW_NORMAL)
        cv2.imshow("Select QR Area", image)
        cv2.setMouseCallback("Select QR Area", select_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(rect) == 2:
            x1, y1 = rect[0]
            x2, y2 = rect[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            with open("coordinates.json", "w") as json_file:
                json.dump({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, json_file, indent=4)
            
            # qr_img = Image.open(qr_path).convert("RGBA")
            if isinstance(qr_path, str):
                qr_img = Image.open(qr_path).convert("RGBA")
            else:
                qr_img = qr_path.convert("RGBA")  # If it's already an image object

            qr_np = np.array(qr_img)
            qr_resized = cv2.resize(qr_np, (x2 - x1, y2 - y1))
            alpha_qr = qr_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_qr
            
            for c in range(3):
                image[y1:y2, x1:x2, c] = (alpha_qr * qr_resized[:, :, c] + alpha_bg * image[y1:y2, x1:x2, c])
            image[y1:y2, x1:x2, 3] = (alpha_qr * 255 + alpha_bg * image[y1:y2, x1:x2, 3])
            
            cv2.imwrite(save_path, image)
            return save_path
        else:
            raise ValueError("Invalid selection. Please select again.")


    # Function to load coordinates from JSON
    def load_coordinates():
        if os.path.exists("qr_coordinates.json"):
            with open("qr_coordinates.json", "r") as file:
                return json.load(file)
        return None

    # Function to save coordinates to JSON
    def save_coordinates(coords):
        with open("qr_coordinates.json", "w") as file:
            json.dump(coords, file)

    # Function to resize image while keeping aspect ratio
    def resize_image(image, max_width=1000, max_height=700):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        if scale < 1:
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    # Function to allow OpenCV rectangle selection
    def select_qr_position(image_path):
        image = cv2.imread(image_path)
        display_img = image.copy()
        
        global rect_start, rect_end, selecting
        rect_start, rect_end, selecting = None, None, False

        def onselect(event, x, y, flags, param):
            global rect_start, rect_end, selecting
            if event == cv2.EVENT_LBUTTONDOWN:
                rect_start = (x, y)
                selecting = True
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                temp_img = display_img.copy()
                cv2.rectangle(temp_img, rect_start, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select QR Code Position", temp_img)
            elif event == cv2.EVENT_LBUTTONUP:
                rect_end = (x, y)
                selecting = False
                cv2.rectangle(display_img, rect_start, rect_end, (0, 255, 0), 2)
                cv2.imshow("Select QR Code Position", display_img)

        cv2.namedWindow("Select QR Code Position", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select QR Code Position", onselect)
        cv2.imshow("Select QR Code Position", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end
            size = max(abs(x2 - x1), abs(y2 - y1))
            coords = {"x": min(x1, x2), "y": min(y1, y2), "size": size}
            return coords
        return None
    # Function to generate and place QR code **correctly**
    def add_qr_to_image(image_path, qr_url, logo_path, coords):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None or coords is None:
            return None

        qr_img_pil = Certification.generate_qr_with_logo(qr_url, logo_path)
        qr_img_np = np.array(qr_img_pil)
        qr_bgra = cv2.cvtColor(qr_img_np, cv2.COLOR_RGBA2BGRA)
        qr_resized = cv2.resize(qr_bgra, (coords["size"], coords["size"]))

        x, y = coords["x"], coords["y"]
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        roi = image[y:y+coords["size"], x:x+coords["size"]]
        alpha_qr = qr_resized[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_qr

        for c in range(3):  
            roi[:, :, c] = (alpha_qr * qr_resized[:, :, c] + alpha_bg * roi[:, :, c]).astype(np.uint8)

        image[y:y+coords["size"], x:x+coords["size"]] = roi
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    


# Page Configuration
st.set_page_config(page_title="Teranis 2025 - Admin Panel", layout="wide")

# Sidebar - Navigation Menu
st.sidebar.title("📌 Navigation")
menu_options = ["🏠 Home", "📂 Generate Each Event JSON", "📑 Merge JSON", "Unique Code - Certification", "Cerification - JSON Combiner", "Bulk Certifications Renamer (Sl Num -> Unique Code)", "QR Code Insertion"]
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
    event_name = st.text_input("Event Name", value="")
    event_date = st.text_input("Event Date", value="")

    certification_type = st.selectbox(
        "Select the type of certification:", 
        ["Certificate of Participation", "Certificate of Merit", "Certificate of Coordination"]
    )
    notes = st.text_input("Any Notes", value="")
    if st.button("Generate Unique Codes"):
        if uploaded_file is None:
            st.warning("Please upload an Excel file first.")
        elif len(prefix) != 3:
            st.warning("Prefix must be exactly 3 letters.")
        else:
            # Process the file
            df, json_data = Certification.process_excel(uploaded_file, prefix, event_name=str(event_name), event_date = str(event_date), certificate_type=certification_type, notes=notes)

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

elif selected_option == "Cerification - JSON Combiner":
    st.write("Upload multiple JSON files, and they will be merged and saved.")

    uploaded_files = st.file_uploader("Upload JSON files", type="json", accept_multiple_files=True)

    if uploaded_files:
        combined_json = Certification.combine_json_files(uploaded_files)

        if combined_json:
            st.success("JSON files combined successfully!")

            # Ask user for filename
            filename = st.text_input("Enter filename (without extension)", "combined")

            if st.button("Save JSON"):
                output_path = os.path.join("Certifications/", f"{filename}.json")

                # Save the combined JSON to the specified directory
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(combined_json, f, indent=4)

                st.success(f"File saved at: `{output_path}`")

elif selected_option == "Bulk Certifications Renamer (Sl Num -> Unique Code)":
    st.title("Certificate Renamer 📜")

    # Upload Excel File
    excel_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    # Upload ZIP Folder
    zip_file = st.file_uploader("Upload Certificates Folder (ZIP)", type=["zip"])

    if excel_file and zip_file:
        # Save the uploaded Excel file
        excel_path = "uploaded_excel.xlsx"
        Certification.save_uploaded_file(excel_file, excel_path)

        # Extract ZIP folder
        Certification.extract_zip(zip_file)

        # Process Certificates
        if st.button("Process Certificates"):
            result = Certification.process_certificates(excel_path)
            if isinstance(result, list):
                st.success("Renaming Completed!")
                st.write("\n".join(result))

                # Zip processed folder for download
                zip_path = Certification.zip_folder("Certificates")
                with open(zip_path, "rb") as f:
                    st.download_button("Download Processed Certificates", f, file_name="Processed_Certificates.zip")
            else:
                st.error(result)

elif selected_option == "QR Code Insertion":
    st.title("QR Code Certificate Generator")
    uploaded_zip = st.file_uploader("Upload ZIP file containing certificates", type=["zip"])

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getvalue())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
                image_files = [os.path.join(temp_dir, f) for f in zip_ref.namelist() if f.endswith((".png", ".jpg", ".jpeg"))]

            if image_files:
                coordinates = Certification.select_qr_position(image_files[0])

                output_zip_path = os.path.join(temp_dir, "certificates_with_qr.zip")
                with zipfile.ZipFile(output_zip_path, "w") as output_zip:
                    for img_file in image_files:
                        filename = os.path.splitext(os.path.basename(img_file))[0]
                        qr_url = f"https://www.teranis.in/verify?uc={filename}"

                        output_img = Certification.add_qr_to_image(img_file, qr_url, coords=coordinates, logo_path="logo.png")

                        output_img_path = os.path.join(temp_dir, f"{filename}.png")
                        if output_img is not None:
                            cv2.imwrite(output_img_path, output_img)
                            output_zip.write(output_img_path, os.path.basename(output_img_path))

                        # cv2.imwrite(output_img_path, output_img)

                        # output_zip.write(output_img_path, os.path.basename(output_img_path))

                with open(output_zip_path, "rb") as f:
                    st.download_button("Download Certificates with QR", f, "certificates_with_qr.zip", "application/zip")




# Footer
st.markdown("---")
st.markdown("© **Teranis 2025 - LBS College of Engineering**")