import pickle

train_instruct_dict = {}
train_instruct_dict['Go'] = ('respond in the direction of the stimulus', 
                            'respond with the same direction as the displayed orientation', 
                            'choose the orientation displayed', 
                            'respond with the same orientation', 
                            'go in the presented direction', 
                            'select the displayed orientation', 
                            'respond with the identical orientation', 
                            'copy the direction displayed', 
                            'select the direction of presented stimulus', 
                            'select the direction of the stimulus', 
                            'respond with the same orientation as the presented stimulus', 
                            'go in the orientation indicated by the stimulus', 
                            'go in the direction presented on the display', 
                            'respond with the identical orientation', 
                            'opt for the same orientation as indicated by the stimulus')

train_instruct_dict['RT_Go'] = ('respond in direction of the stimulus immediately', 
                                'respond with the displayed orientation at stimulus onset', 
                                'choose the orientation displayed immediately', 
                                'respond with the same direction immediately', 
                                'go in the same direction immediately', 
                                'go in the displayed direction at as soon as the stimulus appears', 
                                'respond with the identical orientation immediately', 
                                'select the orientation displayed at stimulus onset', 
                                'choose the orientation displayed as soon as the stimulus is shown', 
                                'choose the same direction as the stimulus immediately', 
                                'pick the identical direction displayed as soon as it appears', 
                                'as soon as the stimulus appears select the same direction', 
                                'when the stimulus is shown respond in the equivalent direction', 
                                'choose the direction that appears on the display immediately', 
                                'select the presented direction when the stimulus is shown')

train_instruct_dict['Anti_Go'] = ('respond in the opposite orientation of stimulus', 
                                    'respond in the reverse direction of stimulus', 
                                    'choose the opposite of the displayed direction', 
                                    'choose the converse direction', 
                                    'respond with the converse orientation', 
                                    'go in inverse of the displayed orientation', 
                                    'select the converse of the displayed orientation', 
                                    'select the reverse of the displayed orientation', 
                                    'respond in the opposite of the displayed stimulus', 
                                    'choose the inverse of the presented direction', 
                                    'pick the opposite of direction indicated by the stimulus', 
                                    'select the inverse of presented stimulus', 
                                    'pick the reverse of the stimulus displayed', 
                                    'select the opposite of the displayed direction',
                                    'go in the direction opposite of the displayed orientation')

train_instruct_dict['Anti_RT_Go'] = ('respond opposite of the stimulus immediately', 
                                    'respond in the reverse direction of stimulus as soon as the stimulus appears', 
                                    'choose the opposite of the displayed direction at stimulus onset', 
                                    'select the converse orientation immediately after the stimulus is shown', 
                                    'respond with the converse direction immediately', 
                                    'go in the inverse of the displayed orientation as soon as the stimulus appears', 
                                    'choose the converse direction immediately', 
                                    'respond with the opposite of the displayed orientation immediately', 
                                    'go in the reverse of the direction displayed as soon as stimulus appears', 
                                    'as soon as stimulus appears respond in the opposite direction', 
                                    'select the reverse orientation at stimulus onset', 
                                    'choose the inverse orientation of the one displayed at stimulus onset', 
                                    'respond in the converse of the displayed direction as soon as it appears', 
                                    'at stimulus onset go in the reverse direction', 
                                    'select the reverse of the displayed direction at stimulus onset')


train_instruct_dict['DelayGo'] = ('remember the stimulus through the delay and go in the same direction', 
                                    'maintain the displayed direction and respond in the identical direction', 
                                    'during the delay maintain the presented orientation and then select the same direction', 
                                    'remember the orientation and select that direction', 
                                    'when the stimulus is displayed remember it through the delay then respond in the same direction', 
                                    'when the orientation appears maintain it during the delay and then select the identical direction', 
                                    'maintain the stimulus in memory and pick the identical direction',
                                    'remember the orientation during the delay and choose the same direction', 
                                    'during the delay remember the stimulus and respond in with the same orientation', 
                                    'remember the presented direction and select the identical direction after the delay', 
                                    'when the direction is displayed remember it and then respond in the same direction', 
                                    'maintain the stimulus orientation through the delay and select the identical orientation', 
                                    'remember the displayed direction and select the same orientation',
                                    'during the delay maintain the presented orientation and select the identical direction',
                                    'remember the displayed direction through the delay and opt for the same direction')

train_instruct_dict['Anti_DelayGo'] = ('remember the stimulus through the delay and select the opposite direction', 
                                        'maintain the orientation during the delay and opt for the inverse of the displayed direction', 
                                        'maintain the stimulus through the delay and go in the converse direction', 
                                        'when the stimulus appears remember it and choose the opposite orientation', 
                                        'when the direction is displayed remember it and respond with the inverse orientation', 
                                        'remember the stimulus and then select the opposite orientation', 
                                        'during the delay maintain the stimulus and respond with the reverse orientation',
                                        'remember the stimulus and go in the reverse of the displayed direction', 
                                        'during the delay remember the stimulus and choose the opposite direction', 
                                        'remember the orientation and respond with the reverse direction', 
                                        'remember the displayed direction the go in the converse direction', 
                                        'when the stimulus appears remember it and select the opposite of the orientation', 
                                        'during the delay maintain the displayed direction and then choose the inverse orientation', 
                                        'when the orientation is presented remember it and opt for the reverse direction',
                                        'during the delay remember the stimulus and go in the inverse direction'
                                        )

train_instruct_dict['Go_Mod1'] = ('respond to the direction displayed in the first modality', 
                                'opt for the orientation of the stimulus that appears in the first modality', 
                                'select the direction corresponding to the stimulus in the first modality', 
                                'attend only to the stimulus in the first modality and respond in that direction', 
                                'focus on the first modality and select the stimulus that appears', 
                                'attend to the first modality and go in the direction of the displayed stimulus', 
                                'select the orientation that appears in the first modality', 
                                'opt for the direction displayed in the first modality', 
                                'focus on the first modality and respond to the orientation displayed',
                                'choose the stimulus presented in the first modality', 
                                'attend to the first modality and pick the displayed direction', 
                                'select the orientation displayed in the first modality', 
                                'focus only on the first modality and choose the shown orientation', 
                                'opt for the direction of the stimulus that appears in the first modality', 
                                'go in the direction that appears in the first modality')

train_instruct_dict['Go_Mod2'] = ('pay attention only to the second modality and respond to the direction displayed there', 
                                'focus on the second modality and select the displayed direction', 
                                'select the stimulus displayed in the second modality', 
                                'choose the orientation that appears in the second modality', 
                                'attend to the second modality and respond to the displayed stimulus', 
                                'opt for the orientation displayed in the second modality', 
                                'pick the direction that appears in the second modality', 
                                'focus only on the second modality and go in the direction of the stimulus displayed there', 
                                'pick the direction that appears in the second modality', 
                                'respond in the direction displayed in the second modality', 
                                'attend only to the second modality and select the displayed orientation', 
                                'opt for the direction of the stimulus displayed in the second modality',
                                'pick the stimulus orientation that appears in the second modality', 
                                'choose the orientation that appears in the second modality', 
                                'focus on the stimulus displayed in the second modality and respond there')

train_instruct_dict['Anti_Go_Mod1'] = ('focus only on the first modality and respond to the opposite of the displayed direction', 
                                    'select the converse of the orientation presented in the first modality', 
                                    'attend to the first modality and respond in the reverse of the displayed direction', 
                                    'choose the opposite of the orientation that appears in the first modality', 
                                    'pay attention only to the first modality and respond in the opposite of the displayed direction', 
                                    'opt for the opposite of the direction displayed in the first modality', 
                                    'attend to the stimulus in the first modality and go in the converse direction', 
                                    'select the opposite of the direction that appears in the first modality', 
                                    'choose the inverse of the orientation that is shown in the first modality',
                                    'focus on the first modality and respond in the opposite direction of the stimulus displayed there', 
                                    'focus on the first modality and select the converse of the displayed stimulus', 
                                    'opt for the inverse of the stimulus orientation displayed in the first stimulus',
                                    'choose the opposite of the stimulus orientation that appears in the first modality', 
                                    'attend to the first modality and pick the reverse of the direction displayed there', 
                                    'pay attention only to the first modality and choose the inverse of the direction displayed')
 
train_instruct_dict['Anti_Go_Mod2'] = ('pay attention only to the second modality and respond in the inverse direction', 
                                    'respond to the converse of the orientation displayed in the second modality', 
                                    'select the opposite of the direction displayed in the second modality', 
                                    'attend to the stimulus in the second modality and respond in the opposite direction', 
                                    'focus on the second modality and respond in the converse of the direction that appears there', 
                                    'choose the opposite of the stimulus presented in the second modality', 
                                    'opt for the opposite of the orientation presented in the second modality', 
                                    'respond in the reverse of the stimulus orientation from the second modality', 
                                    'pick the reverse of the stimulus displayed in the second modality', 
                                    'attend to the second modality and select the opposite of the direction displayed', 
                                    'focus on the second modality and choose the inverse of the displayed direction', 
                                    'select the opposite of the orientation displayed in the second modality',
                                    'choose the converse of the orientation displayed in the second modality', 
                                    'attend only to the second modality and pick the reverse of the presented orientation',
                                    'choose the opposite of the direction of the stimulus presented in the second modality')
                                

train_instruct_dict['DM'] = ('respond in the direction of highest intensity', 
                            'choose the strongest stimulus',   
                            'go in the direction of the stimulus with maximal strength', 
                            'respond in the direction of the strongest stimulus', 
                            'choose the most intense stimulus',
                            'respond in direction of greatest stimulus strength', 
                            'choose the stimulus presented with highest intensity', 
                            'go in direction of greatest intensity', 
                            'choose the orientation with largest strength', 
                            'select the direction presented with highest intensity', 
                            'respond to the stimulus with maximal intensity', 
                            'select the stimulus orientation with greatest strength', 
                            'select the stimulus with greatest intensity', 
                            'respond with the orientation presented with greatest strength',
                            'choose the most intensely displayed orientation')

train_instruct_dict['Anti_DM'] = ('respond in the direction of minimal strength', 
                                    'choose the weakest stimulus', 
                                    'go in the direction of stimulus with least intensity', 
                                    'select the stimulus presented with the lowest strength', 
                                    'go in the direction presented with lowest intensity', 
                                    'choose the stimulus with minimal strength', 
                                    'respond to the direction presented with the weakest strength', 
                                    'pick the stimulus with least strength', 
                                    'select the orientation presented with minimal strength', 
                                    'choose the stimulus with lowest intensity', 
                                    'go in the direction presented with weakest intensity', 
                                    'pick the weakest direction',
                                    'select the orientation with lowest intensity', 
                                    'pick the least intense direction', 
                                    'respond in the direction of minimal intensity') 

train_instruct_dict['MultiDM'] = ('respond to the combined greatest value between two stimuli', 
                                    'choose the direction with highest average intensity between two modalities', 
                                    'select the direction with highest overall strength between two stimuli', 
                                    'respond in the direction of highest combined stimulus strength', 
                                    'go in the direction with largest joint intensity between stimuli', 
                                    'select the orientation with highest average strength over both modalities', 
                                    'go in the direction of highest combined stimulus value', 
                                    'choose the direction representing the highest integrated stimulus strength', 
                                    'respond in the direction with greatest sum over two displayed stimulus values', 
                                    'select the direction with greatest average value across modalities',
                                    'choose the orientation with highest intensity integrated over modalities', 
                                    'respond with the orientation displayed with maximal strength averaged across modalities',
                                    'go in the direction of stimulus with maximal intensity over both modalities',
                                    'pick the orientation with greatest value combined over modalities',
                                    'pick the orientation presented with highest average intensity')

train_instruct_dict['Anti_MultiDM'] = ('respond in the direction with lowest combined value between two stimuli',
                                        'select the orientation with least average intensity between two modalities', 
                                        'select the direction with weakest average value between two stimuli', 
                                        'respond in the direction of minimal combined stimulus strength', 
                                        'go in the direction of smallest intensity integrated over both two modalities', 
                                        'choose the orientation with lowest average value over both modalities', 
                                        'choose the orientation which has the weakest joint intensity over modalities', 
                                        'respond in the direction which has the lowest combined intensity', 
                                        'select the orientation with the minimal strength over modalities', 
                                        'go in the direction with weakest average value across modalities',
                                        'pick the direction displayed with least intensity across modalities',
                                        'pick the stimulus displayed with the lowest combined strength between modalities',
                                        'choose the stimulus presented with weakest intensity averaged over modalities',
                                        'respond in the direction with minimal strength across both modalities',
                                        'select the direction which has the least combined strength')

train_instruct_dict['ConDM'] = ('respond to the the strongest stimulus only if you are confident otherwise do not respond', 
                                'if you are sure of the correct answer respond to the strongest stimulus otherwise do not respond', 
                                'select the orientation displayed with greatest strength if you have high confidence otherwise do not respond', 
                                'choose the direction with highest intensity if you are confident in your decision otherwise do not respond', 
                                'if you are confident about the answer respond to the orientation with maximal intensity otherwise do not respond', 
                                'go in the direction of greatest stimulus strength if you have high confidence in your answer otherwise do not respond', 
                                'respond to the stimulus presented with highest intensity if you are sure of your decision otherwise do not respond', 
                                'if you are sure of the result select the stronger direction otherwise do not respond', 
                                'if you are confident about the result pick the most intense stimulus otherwise do not respond', 
                                'pick the orientation that appears with greatest strength if you have high confidence otherwise do not respond', 
                                'opt for the direction displayed with highest intensity if you are sure of your answer otherwise do not respond', 
                                'if you are confident in your answer select the stimulus with highest intensity otherwise do not respond', 
                                'opt for the most intensely displayed direction if you are sure about your decision otherwise do not respond', 
                                'respond in the direction of greatest strength if you are confident otherwise do not respond', 
                                'choose the strongest orientation if you have high confidence otherwise do not respond')

train_instruct_dict['Anti_ConDM'] = ('opt for the stimulus presented with least strength if you are confident in your answer otherwise do not respond', 
                                    'if you have high confidence select the stimulus displayed with minimal intensity otherwise do not respond',
                                    'if you are confident in your answer pick the weakest stimulus otherwise do not respond', 
                                    'choose the orientation that appears with lowest intensity if you are confident otherwise do not respond', 
                                    'pick the weakest direction if you are confident in your answer otherwise do not respond', 
                                    'respond to the stimulus with minimal strength if you are sure about your answer otherwise do not respond', 
                                    'go in the direction of the stimulus presented with least strength if you are sure about your decision otherwise do not respond', 
                                    'if you have high confidence choose the orientation with least intensity otherwise do not respond', 
                                    'if you are sure about your answer select the direction with least strength otherwise do not respond', 
                                    'pick the stimulus which appears with minimal strength if you are confident in your answer otherwise do not respond', 
                                    'select the least intense direction if you have high confidence otherwise do not respond', 
                                    'choose the stimulus presented with least strength if you are confident in your decision otherwise do not respond', 
                                    'respond to the orientation with lowest strength if you are sure of your answer otherwise do not respond', 
                                    'if you have high confidence choose the orientation with least strength otherwise do not respond', 
                                    'opt for the weakest direction if you have high confidence otherwise do not respond'
                                    )  

train_instruct_dict['ConMultiDM'] = ('respond to the stimuli presented with greatest strength integrated across modalities if you are confident in your decision otherwise do not respond', 
                                    'if you are confident in your answer respond to the stimulus with highest intensity over both modalities otherwise do not respond', 
                                    'select the direction displayed with highest intensity over both modalities if you are confident otherwise do not respond', 
                                    'choose the direction with highest intensity across both modalities if you have high confidence otherwise do not respond', 
                                    'if you are confident pick the orientation displayed with highest value combined over modalities otherwise do not respond', 
                                    'if you are sure of your answer choose the direction with largest integrated value over both modalities otherwise do not respond', 
                                    'opt for the orientation with highest average intensity over modalities if you have high confidence otherwise do not respond', 
                                    'select the direction with greatest average value across modalities if you are sure of your answer otherwise do not respond', 
                                    'respond in the direction of greatest combined strength if you have high confidence in your answer otherwise do not respond', 
                                    'opt for the orientation with maximal intensity over modalities if you are confident otherwise do not respond', 
                                    'if you are confident in your answer select the stimulus with highest combined stimulus strength otherwise do not respond', 
                                    'if you have high confidence pick the stimulus with highest intensity integrated over modalities otherwise do not respond', 
                                    'choose the direction with largest joint intensity between modalities if you are sure about your decision otherwise do not respond', 
                                    'pick the stimulus with maximal intensity over modalities if you have high confidence otherwise do not respond', 
                                    'select the orientation with greatest combined strength over modalities if you are confident otherwise do not respond')

train_instruct_dict['Anti_ConMultiDM'] = ('opt for the stimulus with least combined strength over modalities if you are confident in your answer otherwise do not respond', 
                                        'choose the orientation presented with minimal strength over both modalities if you have high confidence otherwise do not respond', 
                                        'select the direction displayed with lowest combined strength over modalities if you are confident in your decision otherwise do not respond', 
                                        'opt for the direction with least combined strength over modalities if you have high confidence otherwise do not respond', 
                                        'select the orientation that appears weakest when integrated over both modalities if you have confidence in your answer otherwise do not respond', 
                                        'if you have high confidence respond to the stimulus with lowest combined strength otherwise do not respond', 
                                        'if you are confident in your decision then select the orientation with minimal intensity over both modalities otherwise do not respond', 
                                        'if you have high confidence pick the stimulus displayed with minimal strength averaged over both modalities otherwise do not respond', 
                                        'pick the orientation with lowest combined strength over modalities if you are confident otherwise do not respond', 
                                        'respond in the direction displayed with least intensity over both modalities if you are confident in your decision otherwise do not respond', 
                                        'choose the weakest direction averaged over both modalities if you have high confidence otherwise do not respond', 
                                        'opt for the stimulus that appears with minimal strength integrated over both modalities if you are confident otherwise do not respond', 
                                        'if you are confident in your answer respond to the orientation presented with least strength over modalities otherwise do not respond', 
                                        'if you have high confidence then select the stimulus with least intensity over modalities otherwise do not respond', 
                                        'pick the weakest direction averaged over modalities if you have confidence in your decision otherwise do not respond')

train_instruct_dict['RT_DM'] = ('respond to the strongest direction as soon as the stimulus appears',
                                'immediately select the direction with greatest strength', 
                                'choose the direction displayed with greatest strength as soon as the stimuli appears', 
                                'select the orientation displayed with highest intensity as soon as the stimuli appears', 
                                'respond to the strongest stimulus as soon as they are displayed', 
                                'opt for the most intensely displayed stimulus as soon as they appear', 
                                'choose the direction displayed with greatest intensity at stimulus onset', 
                                'pick the orientation which has most strength immediately', 
                                'immediately select the direction with maximal strength', 
                                'respond to the stimulus that is presented with maximal intensity as soon as the stimuli appears', 
                                'opt for the orientation with largest strength immediately', 
                                'immediately select the strongest stimulus', 
                                'pick the most intensely displayed direction as soon as the stimulus appears', 
                                'at stimulus onset respond to the direction displayed with greatest strength',
                                'pick the orientation presented with most strength at stimulus onset')

train_instruct_dict['Anti_RT_DM'] = ('at stimulus onset select the weakest stimulus',
                                    'opt for the stimulus displayed with minimal strength as soon as stimulus appears', 
                                    'pick the orientation that is presented with least intensity immediately', 
                                    'at stimulus onset choose the stimulus with lowest intensity', 
                                    'respond to the direction that is displayed with minimal intensity as soon as the stimuli appears', 
                                    'pick the least intense direction immediately', 
                                    'select the stimulus with least intensity as soon as the stimulus appears', 
                                    'at stimulus onset choose the orientation with lowest intensity',
                                    'respond to the stimulus that is displayed with minimal intensity as soon as the stimuli appear', 
                                    'immediately opt for the orientation that has least strength', 
                                    'respond to the orientation that is least intense as soon as the stimuli appear', 
                                    'at stimulus onset pick the weakest direction',
                                    'select the stimulus that appears with lowest intensity as soon as the stimuli appear',
                                    'go in the direction of the stimuli presented least intensely immediately',
                                    'at stimulus onset choose the stimulus with weakest intensity')    

train_instruct_dict['DelayDM'] = ('remember both directions and respond to the strongest one', 
                                'maintain both stimuli in memory and select the most intense one', 
                                'remember both orientations and respond to the one displayed with highest intensity', 
                                'maintain each direction in memory and choose the direction that was presented with greatest strength', 
                                'remember each displayed orientation through the delay and select the strongest one',
                                'during the delay remember the first stimulus and then respond to the direction presented with maximal strength',
                                'remember both stimuli and pick the most intense one', 
                                'remember each direction and opt for the strongest one', 
                                'maintain each orientation in memory and choose the one that was displayed with highest intensity', 
                                'maintain the stimulus through the delay and choose the one that was displayed more intensely', 
                                'remember each stimulus and respond to the strongest one',
                                'remember both orientations and choose the one that appeared with most strength', 
                                'during the delay remember the first orientation and then select the orientation displayed most intensely',
                                'remember each displayed direction and opt for the one with maximal strength',
                                'maintain each stimulus in memory and pick the one with maximal strength')

train_instruct_dict['Anti_DelayDM'] = ('maintain both stimuli in memory and choose the one with least intensity', 
                                    'remember each stimuli and then respond to the direction presented with lowest intensity', 
                                    'during the delay remember the first orientation and then choose the least intense orientation', 
                                    'remember both directions and pick the weakest one', 
                                    'remember each orientation and select the one that appears with minimal intensity', 
                                    'maintain both stimuli in memory and pick the one that has weakest intensity', 
                                    'remember both directions and select the one with least strength', 
                                    'during the delay remember each orientation and opt for the weakest one', 
                                    'remember each stimulus and respond to the one with minimal strength', 
                                    'maintain each displayed direction and pick the one presented with lowest strength', 
                                    'remember both displayed directions and then choose the one that is least intense', 
                                    'maintain both orientations in memory and choose the one presented least intensely',
                                    'remember each stimuli then go in the direction presented with lowest strength',
                                    'remember both stimuli and then respond to the orientation with least intensity',
                                    'remember the displayed directions through the delay then choose the weakest one')

train_instruct_dict['DelayMultiDM'] = ('maintain stimuli in memory through the delay and select the direction that has highest average strength over modalities',
                                        'remember both stimuli and select the direction displayed with highest joint intensity over both modalities', 
                                        'remember each orientation and pick the one that has greatest strength over both modalities', 
                                        'during the delay remember each stimulus and then choose the one which is displayed with highest intensity averaged over both modalities', 
                                        'remember each stimuli through the delay and then respond to the most intense direction across both modalities', 
                                        'maintain stimuli in memory through the delay and then opt for the orientation which is the strongest over both modalities', 
                                        'remember each stimuli and choose the direction displayed most intensely over both modalities', 
                                        'during the delay remember each direction and pick the one with highest joint intensity over modalities', 
                                        'remember each direction then select the one with greatest average strength', 
                                        'remember both orientations and respond to the one with highest value integrated over both modalities', 
                                        'maintain each stimuli in memory and select the one with maximal strength combined over both modalities',
                                        'remember each direction and respond to the one with maximal intensity averaged over both modalities',
                                        'remember both stimuli through the delay and then pick the one with greatest strength combined over both modalities',
                                        'maintain the stimuli in memory and respond to the direction that was presented with highest intensity integrated over both modalities',
                                        'remember both orientations and select the one that is most intensely displayed across both modalities')

train_instruct_dict['Anti_DelayMultiDM'] = ('maintain both directions in memory through the delay and choose the stimulus presented with least intensity over modalities', 
                                            'remember each orientation and choose the one that is weakest averaged over both modalities', 
                                            'during the delay remember each orientation and then select the one presented with least strength over modalities',
                                            'remember both directions and then choose the one that has minimal strength combined over both modalities' , 
                                            'during the delay maintain each orientation in memory and opt for the one with least joint intensity over both modalities', 
                                            'remember each orientation and then choose the one with lowest combined strength over both modalities', 
                                            'maintain each direction in memory and then pick the direction which was presented with minimal strength averaged over both modalities', 
                                            'remember both directions and respond to the one which has lowest intensity combined over both modalities', 
                                            'during the delay maintain both stimuli in memory and then opt for the one with lowest overall strength over both modalities',
                                            'maintain both stimuli in memory and select the direction displayed with weakest intensity averaged over both modalities', 
                                            'remember both stimuli through the delay and go in the direction of the one with minimal joint strength over both modalities', 
                                            'remember each stimulus and choose the one presented with lowest strength integrated over both modalities', 
                                            'maintain both stimuli in memory and select the one which has weakest intensity when averaged over both modalities',
                                            'remember both directions and pick the stimulus with lowest combined strength over both modalities',
                                            'remember each stimulus and respond in the direction which is presented with minimal strength averaged over both modalities')

train_instruct_dict['DM_Mod1'] = ('select the orientation in the first modality that is strongest', 
                                'choose the most intense stimulus that appears in the first modality', 
                                'attend to the first modality and select the orientation displayed with highest intensity', 
                                'focus only on the first modality and choose the direction with greatest strength', 
                                'opt for the direction with largest strength in the first modality', 
                                'select the stimulus with maximal intensity in the first modality', 
                                'attend only to the first modality and choose the strongest direction', 
                                'respond to the stimulus displayed with greatest strength in the first modality', 
                                'attend only to the first modality and pick the orientation displayed with maximal strength', 
                                'choose the stimulus which appears with highest intensity in the first modality', 
                                'select the most intensely displayed direction in the first modality',
                                'respond to the direction in the first modality that is strongest', 
                                'attend only to the first modality and opt for the direction with highest intensity', 
                                'focus on the first modality and choose the orientation which has greatest strength', 
                                'select the direction in the first modality displayed most intensely')

train_instruct_dict['DM_Mod2'] = ('choose the direction in the second modality that appears strongest', 
                                'attend only to the second modality and respond to the most intense orientation', 
                                'focus on the second modality and select the direction displayed with greatest strength',
                                'pick the stimulus in the second modality that is displayed with maximal intensity', 
                                'opt for the direction displayed in the second modality with most intensity', 
                                'select the stimulus that has maximal strength in the second modality', 
                                'focus only on the second modality and choose the strongest direction that appears there', 
                                'respond to the direction which appears with highest intensity in the second modality', 
                                'attend only to the second modality and pick the orientation that appears strongest', 
                                'attend only to the second modality and opt for the stimulus displayed there with greatest strength', 
                                'select the stimulus in the second modality which has maximal strength', 
                                'pick the direction that is displayed most intensely in the second modality', 
                                'focus on the second modality and respond to the orientation displayed there with highest intensity',
                                'choose the stimulus in the second modality which has maximal strength',
                                'pick the direction that is most intense in the second modality')

train_instruct_dict['Anti_DM_Mod1'] = ('attend to the first modality and choose the weakest direction', 
                                    'attend to the first modality and respond in the direction of minimal intensity',
                                    'select the orientation in the first modality which has lowest strength', 
                                    'pick the stimulus in the first modality that is presented with least intensity', 
                                    'focus only on the first modality and choose the direction that appears weakest', 
                                    'respond to the stimulus in the first modality that has lowest strength', 
                                    'choose the direction in the first modality which is weakest', 
                                    'select the stimulus in the second modality that has minimal strength', 
                                    'attend to the first modality and select the direction that is displayed with least strength', 
                                    'opt for the direction in the first modality that appears weakest',
                                    'focus only on the first modality and choose the orientation displayed with minimal intensity', 
                                    'select the stimulus in the first modality that has lowest intensity', 
                                    'pick the direction which appears with minimal intensity in the first modality',
                                    'opt for the stimulus that appears weakest in the first modality',
                                    'focus on the first modality and choose the stimulus minimal strength')

train_instruct_dict['Anti_DM_Mod2'] = ('attend to the second modality and choose the orientation that appears weakest', 
                                    'focus on the second modality and respond to the least intense direction',
                                    'choose the direction with least intensity in the second modality',
                                    'attend to the second modality and select the direction with least strength', 
                                    'choose the least intense orientation in the second modality', 
                                    'go in the direction of the weakest stimulus in the second modality', 
                                    'respond to the orientation which has lowest strength in the second modality', 
                                    'focus only on the second modality and pick the stimulus with lowest intensity', 
                                    'attend to the second modality and opt for the weakest stimulus presented there', 
                                    'pick the direction in the second modality presented with least strength', 
                                    'focus on the second modality and select the weakest direction', 
                                    'select the direction with lowest strength in the second modality', 
                                    'choose the direction with weakest intensity in the second modality',
                                    'attend to the second modality and choose the orientation with lowest intensity',
                                    'attend only to the second modality and select the direction with lowest strength')


train_instruct_dict['RT_DM_Mod1'] = ('select the direction in the first modality that appears strongest at stimulus onset', 
                                    'attend to the first modality and select the most intense orientation as soon as the stimuli appear', 
                                    'focus only on the first modality and choose the orientation with maximal strength immediately', 
                                    'respond to the stimulus that appears strongest in the first modality as soon as the stimuli appear', 
                                    'choose the highest intensity stimulus in the first modality at stimulus onset', 
                                    'select the stimulus most intense stimulus in the first modality as soon as the stimulus appears',
                                    'attend only to the first modality and opt for the highest strength stimulus at stimulus onset', 
                                    'focus on the first modality and pick the strongest direction as soon as the stimuli appear', 
                                    'focus only on the first modality and choose the orientation with maximal strength immediately', 
                                    'attend to the first modality and respond to the highest intensity stimulus as soon as the stimuli appear', 
                                    'select the strongest orientation in the first modality as stimulus onset', 
                                    'pick the most intense direction displayed in the first modality as stimulus onset', 
                                    'focus on the first modality and opt for the stimulus with maximal intensity as soon as the stimuli appear',
                                    'pick the direction in the first modality with largest strength at stimulus onset',
                                    'choose the orientation in the first modality with maximal strength immediately')

train_instruct_dict['RT_DM_Mod2'] = ('focus only on the second modality and pick the orientation that is strongest at stimulus onset', 
                                    'opt for the direction in the second modality that has highest intensity as soon as the stimuli appear', 
                                    'respond to the orientation in the second modality that has maximal strength immediately', 
                                    'attend only to the second modality and select the direction with greatest strength as soon as the stimuli appear', 
                                    'attend only to the second modality and opt for the orientation with highest strength at stimulus onset', 
                                    'select the stimulus in the second modality that has greatest intensity at stimulus onset', 
                                    'focus only on the second modality and pick the most intensely displayed orientation immediately', 
                                    'choose the direction in the second modality displayed most intensely as soon as the stimuli appear', 
                                    'focus only on the second modality and opt for the highest intensity orientation at stimulus onset', 
                                    'opt for the orientation in the second modality that is strongest as soon as the stimuli appear', 
                                    'attend only to the second modality and select the strongest orientation at stimulus onset',
                                    'attend only to the second modality and choose the stimulus with greatest strength as soon as the stimuli appear',
                                    'pick the direction in the second modality that has largest intensity immediately',
                                    'respond to the orientation in the second modality which has maximal strength at stimulus onset',
                                    'focus on the second modality and go in the direction greatest intensity as soon as the stimuli appear'
                                    )

train_instruct_dict['Anti_RT_DM_Mod1'] = ('focus only on the first modality and select the weakest orientation at stimulus onset', 
                                        'choose the direction in the first modality that appears weakest as soon as the stimuli appear', 
                                        'attend only to the first modality and respond to the orientation with weakest intensity immediately', 
                                        'select the orientation in the first modality that has minimal strength as soon as the stimuli appear', 
                                        'respond to the direction in the first modality that is least intense immediately', 
                                        'attend only to the first modality and go in the direction with lowest strength as soon as the stimuli appear', 
                                        'focus only on the first modality and select the least intense orientation at stimulus onset', 
                                        'focus on the first modality and choose the orientation with minimal strength as soon as the stimuli appear', 
                                        'pick the direction in the first modality that is least intense immediately', 
                                        'respond to the orientation in the first modality that has least strength at stimulus onset', 
                                        'choose the stimulus in the first modality presented with lowest intensity as soon as the stimuli appear', 
                                        'attend to the first modality and pick the stimulus with minimal intensity immediately', 
                                        'attend to the first modality and respond to the weakest orientation at stimulus onset', 
                                        'select the stimulus in the first modality that appears weakest as soon as the stimuli appear', 
                                        'choose the orientation in the first modality presented with minimal strength immediately'
                                        )
train_instruct_dict['Anti_RT_DM_Mod2'] = ('focus on the second modality and select the least intensely displayed direction at stimulus onset', 
                                        'opt for the direction in the second modality that has minimal strength immediately', 
                                        'opt for the orientation in the second modality that is displayed with least strength as soon as the stimuli appear', 
                                        'attend only to the second modality and pick the direction with minimal intensity at stimulus onset', 
                                        'attend only to the second modality and opt for the orientation that is weakest immediately', 
                                        'select the direction in the second modality that has lowest strength as soon as the stimuli appear', 
                                        'respond to the stimulus in the second modality which has lowest strength as soon as the stimulus appears', 
                                        'select the stimulus in the second modality that is least intensely displayed immediately', 
                                        'focus only on the second modality and choose the lowest strength orientation at stimulus onset', 
                                        'focus on the second modality and respond to the least intense stimulus as soon as the stimuli appear', 
                                        'pick the orientation in the second direction that is presented with minimal strength at stimulus onset', 
                                        'go in the direction of the stimulus in the second modality which is weakest immediately', 
                                        'attend to the second modality and choose the weakest direction at stimulus onset', 
                                        'select the direction in the second modality that has minimal strength as soon as the stimuli appear', 
                                        'opt for the stimulus in the second modality which is displayed least intensely at stimulus onset'
                                        )


train_instruct_dict['COMP1'] = ('if the first stimulus is greater than the second stimulus respond to the first stimulus otherwise do not respond', 
                                'if the first stimulus has higher intensity than the second go in the first direction otherwise do not respond', 
                                'go in the direction of the first stimulus if it is stronger than the second stimulus otherwise do not respond', 
                                'when the initial stimulus has higher value than the final stimulus respond in the initial direction otherwise do not respond', 
                                'if the earlier stimulus has greater strength than the later stimulus respond in the first direction otherwise do not respond',   
                                'respond to the initial stimulus if it is stronger than the last stimulus otherwise do not respond', 
                                'choose the earlier stimulus when it is presented with higher intensity relative to the second stimulus otherwise do not respond', 
                                'if the initial stimulus has higher value than the second respond to the first direction otherwise do not respond', 
                                'when the first stimulus is presented with the higher intensity than the second select the first direction otherwise do not respond', 
                                'choose the first stimulus if it is presented with the greater intensity than the second otherwise do not respond',
                                'select the initial stimulus if is stronger than the final stimulus otherwise do not respond',
                                'if the first stimulus has larger value than than the second stimulus select the first direction otherwise do not respond',
                                'respond to the earlier direction if it is more intense than the last stimulus otherwise do not respond',
                                'if the first stimulus is more intense than the second stimulus choose the first direction otherwise do not respond',
                                'when the initial stimulus is presented with greater strength than the second stimulus respond to the first direction otherwise do not respond')

train_instruct_dict['COMP2'] = ('respond in the direction of the second stimulus if it has greater intensity than the first otherwise do not respond', 
                                'if the final stimulus is presented with higher value than the initial stimulus go in the final direction otherwise do not respond', 
                                'when the final stimulus has greater strength than the first select the second orientation otherwise do not respond', 
                                'respond in the direction of the second stimulus if it has higher intensity than the initial stimulus otherwise do not respond', 
                                'if the second stimulus has higher value than the first stimulus choose the second orientation otherwise do not respond', 
                                'select the last stimulus direction if it is presented with greater strength than the first otherwise do not respond', 
                                'when the last stimulus is presented with more intensity than the first respond in the second direction otherwise do not respond', 
                                'choose the final stimulus if it has the greater value than the first otherwise do not respond', 
                                'if the final stimulus is presented with the higher intensity than the initial stimulus respond to the second stimulus otherwise do not respond', 
                                'select the second stimulus if it is presented with the higher intensity than the first stimulus otherwise do not respond',
                                'when the second stimulus is more intense than the first stimulus respond to the second stimulus otherwise do not respond',
                                'if the second stimulus is presented with greater strength than the first respond to the second stimulus otherwise do not respond',
                                'choose the second stimulus if it is more intense than the the first stimulus otherwise do not respond',
                                'when the final stimulus is more intense than the first respond to the second stimulus otherwise do not respond',
                                'pick the second stimulus if it has more intensity than the first otherwise do not respond')

train_instruct_dict['MultiCOMP1'] = ('respond to the first direction when it has greater strength on average than the second stimulus otherwise do not respond',
                                        'respond if the first direction has higher combined intensity over two modalities than the second stimulus otherwise do not respond', 
                                        'if the joint intensity of the first directions is higher than the second then respond to the first direction otherwise do not respond', 
                                        'choose the initial direction if the integrated strength across modalities is greater than the final direction otherwise do not respond', 
                                        'go in the initial direction if it has greater value than the final direction when combined over both modalities otherwise do not respond', 
                                        'select the first direction when it has greater integrated strength across modalities than the second direction otherwise do not respond', 
                                        'if the initial direction has higher overall value over modalities than the final direction then choose the initial direction otherwise do not respond', 
                                        'if the combined intensity of the first direction is higher than the second direction then respond to the first direction otherwise do not respond', 
                                        'choose the initial direction when the joint strength over both modalities is greater than the final direction otherwise do not respond', 
                                        'pick the first direction if the average value is larger than the second direction for both modalities otherwise do not respond', 
                                        'select the initial direction if it displays a higher overall intensity over modalities than the final direction otherwise do not respond',
                                        'if the intensity of the first stimulus integrated over modalities is greater than the second respond in the first direction otherwise do not respond',
                                        'select the first stimulus if it has higher joint strength over modalities than the second otherwise do not respond',
                                        'if the first direction is presented with higher intensity averaged over modalities than the second direction then respond to the first direction otherwise do not respond',
                                        'pick the initial stimulus if it has greater joint intensity over both modalities than the second otherwise do not respond')

train_instruct_dict['MultiCOMP2'] = ('respond in the second direction if it has larger overall strength over modalities than the first direction otherwise do not respond', 
                                        'if the final direction has greater combined value than the initial direction then respond in the final direction otherwise do not respond', 
                                        'respond in the second direction when it has higher integrated value over modalities than the first direction otherwise do not respond', 
                                        'select the second direction if it has greater strength than the first direction over both modalities otherwise do not respond', 
                                        'choose the final direction if it has a joint intensity higher than the initial direction over modalities otherwise do not respond', 
                                        'if the final direction is presented with greater overall strength across modalities than the initial direction choose the final direction otherwise do not respond', 
                                        'respond to the second direction when it has larger combined value over modalities than the first direction otherwise do not respond', 
                                        'select the final direction if the strength integrated over modalities is greater than the first direction otherwise do not respond', 
                                        'if the second direction is represented with higher joint intensity over both modalities than the second then respond to the second direction otherwise do not respond', 
                                        'choose the second direction when its combined strength over modalities is higher than the first direction otherwise do not respond',
                                        'if the second stimulus is presented with higher joint intensity over modalities than the first then select the second stimulus otherwise do not respond',
                                        'pick the final direction if it has larger strength combined over modalities than the first direction otherwise do not respond',
                                        'respond to the second stimulus if it is presented with greater intensity averaged over modalities than the first stimulus otherwise do not respond',
                                        'choose the second direction if it has stronger presentation averaged over modalities than the first direction otherwise do not respond',
                                        'if the final stimulus has higher overall value for both modalities than the initial direction then select the final stimulus otherwise do not respond')

train_instruct_dict['DMS'] = ('if the first and the second stimuli match then respond with that orientation otherwise do not respond', 
                            'if the same directions are displayed then respond to the stimuli otherwise do not respond', 
                            'if the stimuli match respond in the direction of the stimuli otherwise do not respond',
                            'when the two displayed directions are the same respond to the stimulus otherwise do not respond', 
                            'if the stimuli match go in the displayed direction otherwise do not respond', 
                            'select the displayed direction if the directions match otherwise do not respond', 
                            'respond if the stimuli are displayed with the same orientation otherwise do not respond', 
                            'if the first and second stimuli has the same orientation then respond otherwise do not respond', 
                            'when the two stimuli match respond in the displayed direction otherwise do not respond', 
                            'if the stimuli match then respond otherwise do not respond', 
                            'if the displayed directions are identical then select the displayed direction otherwise do not respond',
                            'select the displayed orientation if the first and second stimuli are identical otherwise do not respond',
                            'pick the stimuli direction if the first and second direction are identical otherwise do not respond',
                            'choose the presented orientation if the first and second stimuli match otherwise do not respond',
                            'respond in the displayed direction if the first and second stimuli match otherwise do not respond')

train_instruct_dict['DNMS'] = ('if the first and second stimuli are different then respond to the second stimuli otherwise do not respond', 
                                'if the stimuli are mismatched go in direction of the second stimuli otherwise do not respond', 
                                'when the displayed directions are distinct go in final direction otherwise do not respond', 
                                'when stimuli are presented in different directions respond to the second direction otherwise do not respond', 
                                'respond in the last direction if stimuli are mismatched otherwise do not respond', 
                                'go in the second direction when stimuli orientations are different otherwise do not respond', 
                                'respond in the final displayed direction if stimuli are mismatched otherwise do not respond', 
                                'if stimuli are mismatched respond in the last direction otherwise do not respond', 
                                'if displayed directions are distinct select the final direction otherwise do not respond', 
                                'go in second displayed direction if stimuli are mismatched otherwise do not respond',
                                'if the first and second stimuli are different then select the final direction otherwise do not respond',
                                'select the second orientation if both presented orientations are distinct otherwise do not respond',
                                'choose the last direction if both displayed directions are mismatched otherwise do not respond',
                                'when the stimuli are distinct then select the final stimuli otherwise do not respond',
                                'respond to the second orientation if the presented orientations are different otherwise do not respond')


train_instruct_dict['DMC'] = ('if the stimuli are on the same half of the display go in first direction otherwise do not respond', 
                            'respond to the initial orientation if stimuli are in the same half of the display otherwise do not respond', 
                            'go in first direction if directions are in same half of display otherwise do not respond',
                            'if the stimuli are in the same half of display respond in the first direction otherwise do not respond', 
                            'if displayed orientations are in the same half respond in the first direction otherwise do not respond', 
                            'respond in the initial direction if the stimuli occur in the same half of the display otherwise do not respond', 
                            'when the stimuli are presented on the same half of the display go in the first direction otherwise do not respond', 
                            'if the stimuli are on the same half of the display choose the first direction otherwise do not respond', 
                            'when the displayed directions are in the same half select the initial direction otherwise do not respond', 
                            'choose the first stimulus when both stimuli are presented on the same half of the display otherwise do not respond', 
                            'select the first orientation if both orientation are in the same half of the display otherwise do not respond',
                            'if the first and second direction are in the same half of the display then respond to the initial direction otherwise do not respond',
                            'select the initial stimulus if the initial and final stimuli are presented on the same half of the display otherwise do not respond',
                            'if the displayed directions are on the same half of the display then respond to the first direction otherwise do not respond',
                            'choose the first direction if both presented stimuli are on the same half of the display otherwise do not respond')

train_instruct_dict['DNMC'] = ('if stimuli are on different halves of display respond in the second direction otherwise do not respond', 
                                'when the stimuli are presented are on opposing sides of the display go in the last direction otherwise do not respond', 
                                'go in the final direction when stimuli are on distinct halves of the display otherwise do not respond', 
                                'if the directions are on different halves select the second direction otherwise do not respond', 
                                'when the stimuli appear in distinct halves choose the last direction otherwise do not respond', 
                                'select the final direction if stimuli are displayed on opposing sides of display otherwise do not respond', 
                                'choose the last direction when stimuli appear on different halves otherwise do not respond', 
                                'if the stimuli are displayed on distinct sides then respond in second direction otherwise do not respond', 
                                'select the final orientation if stimuli are on different sides of display otherwise do not respond', 
                                'respond in the last direction when directions are presented on different halves of the display otherwise do not respond',
                                'if the stimuli are presented on opposing sides of the display then select the second stimulus otherwise do not respond',
                                'if the first and second directions are displayed on opposite halves of the display respond in the final direction otherwise do not respond',
                                'go in the second direction when the stimuli are presented on opposite halves of the display otherwise do not respond',
                                'respond to the last orientation if both are presented on distinct halves otherwise do not respond',
                                'choose the final stimulus if both presented stimuli are on opposite halves of the display otherwise do not respond')


test_instruct_dict = {}
test_instruct_dict['Go'] = ('select the presented direction', 
                                'respond in the direction shown',
                                'respond in the direction displayed', 
                                'respond with the displayed orientation', 
                                'choose the presented direction')

test_instruct_dict['RT Go'] = ('choose the direction shown immediately', 
                                    'select the direction presented immediately', 
                                    'select the displayed orientation as soon as stimulus appears', 
                                    'choose the direction presented immediately', 
                                    'immediately respond to the presented orientation')

test_instruct_dict['Anti Go'] = ('select the opposing direction', 
                                        'choose the opposite of the presented direction', 
                                        'respond with the reverse orientation', 
                                        'go in the opposite of displayed orientation',
                                        'pick the converse of the stimulus orientation')

test_instruct_dict['Anti RT Go'] = ('pick the converse of the stimulus direction immediately', 
                                        'select the reverse of the displayed direction immediately', 
                                        'respond with the inverse orientation as soon as it is shown', 
                                        'choose the opposite of displayed orientation at stimulus onset',
                                        'immediately respond in the opposing direction as is shown')

test_instruct_dict['Go_Mod1'] = ('choose the orientation in the first modality',
                                'opt for the stimulus that appears in the first modality',
                                'attend to the first modality and respond in to the displayed direction',
                                'go in the direction stimulus in the first modality',
                                'select the orientation that is displayed in the first modality')

test_instruct_dict['Go_Mod2'] =('respond to the stimulus in the second modality',
                              'select the direction that is displayed in the second modality',
                              'focus on the second modality and choose the displayed direction',
                              'opt for the orientation in the second modality',
                              'respond in the direction presented in the second modality')

test_instruct_dict['Anti_Go_Mod1'] =('go in the opposite direction of the stimulus presented in the first modality',
                                    'attend to the first modality and choose the converse direction',
                                    'select the opposite of the direction displayed in the first modality',
                                    'pick the direction opposite of the one displayed in the first modality',
                                    'choose the reverse of the direction displayed in the first modality')

test_instruct_dict['Anti_Go_Mod2'] =('attend to the second modality and go in the reverse of the displayed orientation',
                                        'opt for the opposite of the direction in the second modality',
                                       'go in the converse of the direction displayed in the second modality',
                                     'focus on the second modality and respond in the reverse direction' ,
                                     'pick the opposite of the orientation in the second modality')

test_instruct_dict['DM'] = ('select the stimulus of greatest strength', 
                                'choose the direction with maximal stimulus intensity', 
                                'pick the direction with stimulus of greatest strength', 
                                'select the orientation displayed with greatest intensity', 
                                'go in the direction of the most intense stimulus')

test_instruct_dict['Anti_DM'] = ('choose the direction with lowest value', 
                                    'respond in the direction that is presented with least intensity', 
                                    'pick the orientation with the lowest strength', 
                                    'go in the direction with weakest presentation strength',
                                    'select the direction with minimal intensity')

test_instruct_dict['MultiDM'] = ('choose the stimuli with highest intensity averaged over modalities', 
                                    'select the orientation with greatest combined strength over modalities', 
                                    'go in direction with highest joint intensity between stimuli', 
                                    'pick the direction of stimulus with greatest intensity between two modalities',
                                    'select the stimuli with maximal combined strength over both modalities')

test_instruct_dict['Anti_MultiDM'] = ('pick the direction with minimal average value over modalities', 
                                            'select the orientation with lowest combined strength', 
                                            'go in the direction of the stimuli with weakest overall value between modalities', 
                                            'choose the direction with lowest presentation strength over modalities',
                                            'select the orientation with weakest intensity over both modalities')

test_instruct_dict['ConDM']

test_instruct_dict['Anti_ConDM']

test_instruct_dict['ConMultiDM']

test_instruct_dict['Anti_ConMultiDM']

test_instruct_dict['DelayDM']

test_instruct_dict['Anti_DelayDM']

test_instruct_dict['DelayMultiDM']

test_instruct_dict['Anti_DelayMultiDM']

test_instruct_dict['DM_Mod1']

test_instruct_dict['DM_Mod2']

test_instruct_dict['Anti_DM_Mod1']

test_instruct_dict['Anti_DM_Mod2']

test_instruct_dict['COMP1'] = ('when the first stimulus is stronger pick the first orientation otherwise do not respond', 
                                    'choose the initial stimulus if it is the stronger of the two presented stimuli otherwise do not respond', 
                                    'if the initial orientation has higher value than the subsequent select the first orientation otherwise do not respond', 
                                    'pick the first direction if the stimulus presented has the greater value otherwise do not respond',
                                    'when the first stimulus has greater intensity than the second then respond to the first stimulus otherwise do not respond')

test_instruct_dict['COMP2'] = ('if the last stimuli is presented with higher intensity than the second respond in second direction otherwise do not respond', 
                                    'pick the direction of the second stimuli if it has greatest value than the first otherwise do not respond', 
                                    'when the final direction has greater intensity then the first go in that direction otherwise do not respond', 
                                    'choose the subsequent stimulus direction if it has greater strength than the first otherwise do not respond',
                                    'if the second orientation is presented with more intensity than the first respond with the first orientation otherwise do not respond')

test_instruct_dict['MultiCOMP1']=('if the initial direction displays higher averaged value than the second direction then respond to initial direction otherwise do not respond', 
                                        'pick the first orientation if it has greater combined intensity than the second orientation otherwise do not respond', 
                                        'when the first direction has higher intensity averaged over modalities than the second direction then select the first direction otherwise do not respond', 
                                        'choose the initial orientation when it has greater joint value over modalities than the final direction otherwise do not respond',
                                        'if the first orientation has greater joint strength over both modalities than the second then respond with the first orientation otherwise do not respond')
                            
test_instruct_dict['MultiCOMP2']=('if the final orientation displays greater overall value than the initial orientations combined over modalities then select the final orientation otherwise do not respond', 
                                        'pick the second direction if it has a bigger average value over modalities than the first direction otherwise do not respond', 
                                        'choose the subsequent direction if it has higher combined intensity over modalities than the initial direction otherwise do not respond', 
                                        'select the second direction if it has a larger joint strength over modalities than the first direction otherwise do not respond',
                                        'respond to the second orientation if it has greater combined intensity than the first orientation otherwise do not respond')

test_instruct_dict['DMS'] = ('respond if stimuli are presented in the same directions otherwise do not respond', 
                                'go in the displayed direction if the stimuli match otherwise do not respond', 
                                'when displayed directions are the same respond to the stimulus otherwise do not respond', 
                                'go in the direction displayed if the two stimuli match otherwise do not respond',
                                'select the stimuli direction if both directions are the same otherwise do not respond')                 

test_instruct_dict['DNMS'] = ('if displayed directions are different go in the second direction otherwise do not respond', 
                                    'when the stimuli are distinct respond to the final stimulus otherwise do not respond',
                                    'if the orientations are mismatched then go in the last direction otherwise do not respond', 
                                    'if the two stimuli mismatched respond to the second stimulus otherwise do not respond',
                                    'select the second direction if both stimuli are presented in different directions otherwise do not respond' )     

test_instruct_dict['DMC'] = ('go in the first orientation if both are displayed on the same side of the display otherwise do not respond', 
                                    'if the stimuli appear in the same half of the display respond to the initial direction otherwise do not respond', 
                                    'if stimuli are on the same half respond in the initial displayed direction otherwise do not respond', 
                                    'pick the first orientation if both stimuli appear on the same side of the display otherwise do not respond',
                                    'select the first stimulus if both are presented on the same side of the display otherwise do not respond')

test_instruct_dict['DNMC'] = ('pick the second direction if both stimuli are on different sides of the display otherwise do not respond', 
                                    'choose the final displayed orientation if they appear on opposing halves of the display otherwise do not respond', 
                                    'if the directions appear on distinct sides of the display respond to the last direction otherwise do not respond', 
                                    'if the stimuli are on different sides respond to the second direction otherwise do not respond',
                                    'select the final orientation if both stimuli appear on opposite halves of the display otherwise do not respond')                                    


def save_instruct_dicts(path):
	pickle.dump(train_instruct_dict, open(path+'/train_instruct_dict', 'wb'))
	pickle.dump(test_instruct_dict, open(path+'/test_instruct_dict', 'wb'))



	