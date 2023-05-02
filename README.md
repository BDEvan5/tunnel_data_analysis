# tunnel_data_analysis

- **Data**: contains training and test data sets
- **Imgs**: Result images
    - Benjamin: results from Benjamin using corrected test environment.
    - keegan: results from original code
    - keegan clean: results from Benjamin's cleaned up version of Keegan's work

## Notes
- Add loss graphs for NN training
    - Train and test splits
- Add a test on the effect of random seeds
- Add test for LSTM
    -  This means ammending the test to include the windows
- [idea:] Instead of single predictions, predict the whole set of 12 temperatures for an input and the previous hours data.
    - Give the temperatures from the previous hour and then give the solar radiation for the prediction hour.
    - Then train it with all the input samples (think langague model). Then it should be able to improve the accuracy

