# imagefilter

1. Install Python 3.10+

Download from python.org. During install tick "Add Python to PATH". Verify:
python --version

2. Create Virtual Environment (venv)
python -m venv venv

Activate it:
# Windows (PowerShell / CMD)
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

# Run the Streamlit App

Make sure your venv is active, then from the project root:
streamlit run app.py
Streamlit will open automatically at http://localhost:8501

Using the App
```1. Upload a PNG or JPG image from the sidebar (grayscale or color — color auto-converts)
1. Upload a PNG or JPG image from the sidebar (grayscale or color — color auto-converts)
2. Select filter type: Mean, Gaussian, or Laplacian Sharpening

3. Adjust kernel size / sigma / sharpening coefficient with the sliders

4. Click Apply Filter

5. Explore tabs: Comparison, Metrics, Frequency Analysis, Kernel

6. Download the filtered image using the button in the Comparison tab
