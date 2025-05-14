{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "88ac5815-b068-4408-8578-6f47c5836956",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.5.1\n"
     ]
    }
   ],
   "source": [
    "import sklearn\n",
    "print(sklearn.__version__)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "4dc4cba3-8954-4d36-80ac-22ed8fab48c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip uninstall scikit-learn\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "e60c3caf-af55-4b3a-99ee-f870b2534c1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install scikit-learn==1.2.2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "368b1713-02d9-4bfc-9607-53e551d5976a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df_orders = pd.read_csv(\"anemia data from Kaggle.csv\")  # e.g., \"data/anemia_data.csv\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ec6d85c6-3be2-44a7-9bdb-e63ec1215a18",
   "metadata": {},
   "outputs": [],
   "source": [
    "file_path = \"anemia data from Kaggle.csv\"\n",
    "if file_path:\n",
    "    df = pd.read_csv(file_path)\n",
    "else:\n",
    "    print(\"No file path provided!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3e0fbf89-25f1-4422-96e4-6b3badb5f1c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('random_forest_model.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "888f2ae9-dc56-4485-8a96-1d54e8c39e6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Load the trained model\n",
    "with open('random_forest_model.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n",
    "\n",
    "# Preprocessing function\n",
    "def preprocess_data(hemoglobin, gender, mcv):\n",
    "    gender_mapping = {'Male': 0, 'Female': 1}\n",
    "    gender = gender_mapping.get(gender, 0)\n",
    "    hemoglobin = max(hemoglobin, 0)\n",
    "    mcv = max(mcv, 0)\n",
    "    data = {'Gender': [gender], 'Hemoglobin': [hemoglobin], 'MCV': [mcv]}\n",
    "    return pd.DataFrame(data)\n",
    "\n",
    "# Prediction function\n",
    "def predict_anemia(hemoglobin, gender, mcv):\n",
    "    df = preprocess_data(hemoglobin, gender, mcv)\n",
    "    prediction = model.predict(df)\n",
    "    return prediction[0]\n",
    "\n",
    "# Streamlit app\n",
    "def main():\n",
    "    st.set_page_config(page_title=\"Anemia Detection App\", layout=\"centered\")\n",
    "    st.title(\"Anemia Detection Using ML\")\n",
    "\n",
    "    st.subheader(\"About Anemia\")\n",
    "    st.write(\"\"\"\n",
    "    Anemia is a condition where your body lacks enough healthy red blood cells or hemoglobin. \n",
    "    This app helps predict if a person is anemic based on their hemoglobin, MCV, and gender using a trained Random Forest model.\n",
    "    \"\"\")\n",
    "\n",
    "    st.write(\"### Average Ranges for Hemoglobin and MCV\")\n",
    "    st.table(pd.DataFrame({\n",
    "        'Gender': ['Male', 'Female'],\n",
    "        'Hemoglobin': ['12.0 - 15.0 g/dL', '11.5 - 14.5 g/dL'],\n",
    "        'MCV': ['80.0 - 95.0 fL', '82.0 - 98.0 fL']\n",
    "    }))\n",
    "\n",
    "    # User Inputs\n",
    "    gender = st.radio(\"Select Gender\", ['Male', 'Female'])\n",
    "    hemoglobin = st.number_input(\"Enter Hemoglobin (g/dL)\", value=12.0, min_value=0.0)\n",
    "    mcv = st.number_input(\"Enter MCV (fL)\", value=90.0, min_value=0.0)\n",
    "\n",
    "    if st.button(\"Detect\"):\n",
    "        prediction = predict_anemia(hemoglobin, gender, mcv)\n",
    "        result = \"Anemic\" if prediction == 1 else \"Non-Anemic\"\n",
    "        st.success(f\"The person is likely to be: **{result}**\")\n",
    "\n",
    "        if mcv < 80:\n",
    "            st.info(\"Type: Likely Microcytic Anemia\")\n",
    "        elif mcv > 100:\n",
    "            st.info(\"Type: Likely Macrocytic Anemia – Please consult a doctor.\")\n",
    "\n",
    "    st.markdown(\"---\")\n",
    "    st.caption(\"🔍 This prediction is for informational purposes. Always consult a medical professional.\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c753209a-f2d7-4486-af12-65771d5b6b0a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
