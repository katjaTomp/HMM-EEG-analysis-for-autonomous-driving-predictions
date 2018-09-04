Steps to execute this code:
   Run from command line: 
   
   
      1. $ python EegPreprocessor.py
      
      After running the previous script, the pickled files were generated. You can move them to a different way. 
      Either way please define the folder in the Train_models.py,states_events_representation.py, loadData_times.py . 
      
      Next, to train the model run,
      2.$ python Train_models.py #here your can define the electrode which should be used as well as any other hyperparameters concerning the mdoel
      
      Next, to extract the events run,
      3.$ python loadData_times.py
      
      Once you have events files with the model, execute the command below to extract data frames with the data before and after simuli onset
      4.$ python states_events_representation.py 
      
      To obtain the actual results, you need to execute:
      5. $ python results_.py
      
    For more information or clarifications, contact katerina.tompoidi@gmail.com
      
