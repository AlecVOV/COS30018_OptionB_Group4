AlecVOV: Check all the To Do task in my code

GitHub Copilot: I've reviewed your code and identified all the TODO tasks. Here's a comprehensive list of what needs to be implemented:

## **Load Data Section (Lines 32-38)**
1. **Check if data has been saved before**
   - If so, load the saved data
   - If not, save the data into a directory

## **Prepare Data Section (Lines 44-50)**
1. **Check if data has been prepared before**
   - If so, load the saved data
   - If not, save the data into a directory
2. **Use a different price value** (e.g., mid-point of Open & Close)
3. **Change the Prediction days** (currently set to 60)

## **Build the Model Section (Lines 95-101)**
1. **Check if model has been built before**
   - If so, load the saved model
   - If not, save the model into a directory
2. **Change the model to increase accuracy**

## **Model Training Section (Lines 146-156)**
1. **Save the model and reload it**
   - Save the trained model for future use
   - Implement model loading functionality

## **Data Processing Section (Lines 174-180)**
1. **Explain line 174**: `model_inputs = model_inputs.reshape(-1, 1)`
2. **Address ISSUE #2**: Scaler computed on training data may cause out-of-bound values in test data
3. **Implement better data processing**: Split data properly for training and testing

## **Predictions Section (Lines 186-191)**
1. **Explain the 5 lines** of x_test preparation code

## **Plotting Section (Lines 199-206)**
1. **Implement Candlestick charts**
2. **Chart showing High & Lows of the day**
3. **Show chart of next few days (predicted)**

## **Code Analysis Questions**
1. **Find the reason** why the predictor performs poorly (10%-13% error)
2. **Explain the difference** with the better-performing code base
3. **Combine different techniques** (CNN + LSTM) for better prediction

Would you like me to help you implement any of these TODO items? I can start with the most critical ones like data persistence, model saving/loading, or improving the prediction accuracy.