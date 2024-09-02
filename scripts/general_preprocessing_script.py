import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(input_file, output_file):
    # Load the data
    data = pd.read_csv(input_file)

    # Select the category column to be encoded
    category_column = 'Category'  # Adjust if necessary

    # Apply OneHotEncoding with the correct argument
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categories = encoder.fit_transform(data[[category_column]])

    # Create a DataFrame with the new encoded columns
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out([category_column]))

    # Merge the new encoded columns with the original DataFrame
    data = data.drop(category_column, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Ensure the columns align with the model
    expected_columns = ['Category_Candies', 'Category_Cigarettes', 'Category_Drinks', 'Category_Others']
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with value 0 if they don't exist

    # Save the preprocessed DataFrame to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Preprocessing completed. File saved as {output_file}")

# Execute the preprocessing for the new file
preprocess_data('../data/sales-07-08-24.csv', '../data/preprocessed_sales-07-08-24.csv')
