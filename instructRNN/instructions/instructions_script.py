import os
import pickle

train_instruct_dict = {}
train_instruct_dict['Go'] = ('respond in the direction of the stimulus', 
                            'respond with the same direction as the displayed orientation', 
                            'choose the orientation displayed', 
                            'respond with the same orientation', 
                            'go in the presented direction', 
                            'select the displayed orientation', 
                            'pick the displayed orientation', 
                            'copy the direction displayed', 
                            'select the direction of presented stimulus', 
                            'choose the direction of the stimulus', 
                            'respond with the same orientation as the presented stimulus', 
                            'go in the orientation indicated by the stimulus', 
                            'go in the direction presented on the display', 
                            'choose the displayed orientation', 
                            'opt for the same orientation as indicated by the stimulus')

train_instruct_dict['RTGo'] = ('respond in direction of the stimulus immediately', 
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

train_instruct_dict['AntiGo'] = ('respond with the opposite orientation', 
                                    'respond in the reverse direction', 
                                    'choose the opposite of the displayed direction', 
                                    'choose the opposite direction', 
                                    'respond with the reverse orientation', 
                                    'go in reverse of the displayed orientation', 
                                    'opt for the reverse of the presented stimulus', 
                                    'select the reverse of the displayed orientation', 
                                    'respond with the reverse of the displayed stimulus', 
                                    'opt for the opposite of the presented direction', 
                                    'pick the opposite of presented direction', 
                                    'select the reverse of presented stimulus', 
                                    'pick the reverse of the stimulus displayed', 
                                    'select the opposite of the displayed direction',
                                    'go in the opposite direction')

train_instruct_dict['AntiRTGo'] = ('respond with the opposite of the stimulus immediately', 
                                    'opt for the opposite direction at stimulus onset', 
                                    'choose the opposite of the displayed direction at stimulus onset', 
                                    'select the reverse orientation immediately', 
                                    'respond with the reverse direction immediately', 
                                    'go in the reverse of the orientation as soon as the stimulus appears', 
                                    'choose the reverse direction immediately', 
                                    'opt for the opposite of the orientation immediately', 
                                    'go in the reverse of the direction as soon as stimulus appears', 
                                    'as soon as stimulus appears respond in the opposite direction', 
                                    'select the reverse orientation at stimulus onset', 
                                    'choose the reverse orientation of the one displayed at stimulus onset', 
                                    'respond in the converse of the displayed direction as soon as the stimulus appears', 
                                    'at stimulus onset go in the reverse direction', 
                                    'select the reverse of the displayed direction at stimulus onset')


train_instruct_dict['GoMod1'] = ('respond to the direction displayed in the first modality', 
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

train_instruct_dict['GoMod2'] = ('pay attention only to the second modality and respond to the direction displayed there', 
                                'focus on the second modality and select the displayed direction', 
                                'select the stimulus displayed in the second modality', 
                                'respond to the orientation that appears in the second modality', 
                                'attend to the second modality and respond to the displayed stimulus', 
                                'opt for the orientation displayed in the second modality', 
                                'go in the direction that appears in the second modality', 
                                'focus only on the second modality and go in the direction of the stimulus displayed there', 
                                'pick the direction that appears in the second modality', 
                                'respond in the direction displayed in the second modality', 
                                'attend only to the second modality and select the displayed orientation', 
                                'opt for the direction of the stimulus displayed in the second modality',
                                'pick the stimulus orientation that appears in the second modality', 
                                'choose the orientation that appears in the second modality', 
                                'focus on the stimulus displayed in the second modality and respond there')

train_instruct_dict['AntiGoMod1'] = ('focus only on the first modality and respond to the opposite of the displayed direction', 
                                    'select the reverse of the orientation presented in the first modality', 
                                    'attend to the first modality and respond in the reverse of the displayed direction', 
                                    'choose the opposite of the orientation that appears in the first modality', 
                                    'pay attention only to the first modality and respond in the opposite of the displayed direction', 
                                    'opt for the opposite of the direction displayed in the first modality', 
                                    'attend to the stimulus in the first modality and go in the opposite direction', 
                                    'select the opposite of the direction that appears in the first modality', 
                                    'choose the reverse of the orientation that is shown in the first modality',
                                    'focus on the first modality and respond in the opposite direction of the stimulus displayed there', 
                                    'focus on the first modality and select the opposite of the displayed stimulus', 
                                    'opt for the reverse of the stimulus orientation displayed in the first stimulus',
                                    'choose the opposite of the stimulus orientation that appears in the first modality', 
                                    'attend to the first modality and pick the reverse of the direction displayed there', 
                                    'pay attention only to the first modality and choose the reverse of the direction displayed')
 
train_instruct_dict['AntiGoMod2'] = ('pay attention only to the second modality and respond in the reverse direction', 
                                    'respond to the opposite of the orientation displayed in the second modality', 
                                    'select the opposite of the direction displayed in the second modality', 
                                    'attend to the stimulus in the second modality and respond in the opposite direction', 
                                    'focus on the second modality and respond in the opposite of the direction that appears there', 
                                    'choose the opposite of the stimulus presented in the second modality', 
                                    'opt for the opposite of the orientation presented in the second modality', 
                                    'respond in the reverse of the stimulus orientation from the second modality', 
                                    'pick the reverse of the stimulus displayed in the second modality', 
                                    'attend to the second modality and select the opposite of the direction displayed', 
                                    'focus on the second modality and choose the reverse of the displayed direction', 
                                    'select the opposite of the orientation displayed in the second modality',
                                    'choose the opposite of the orientation displayed in the second modality', 
                                    'attend only to the second modality and pick the reverse of the presented orientation',
                                    'choose the opposite of the direction of the stimulus presented in the second modality')
                                
train_instruct_dict['RTGoMod1'] = ('focus on the first modality and select the stimulus immediately', 
									'attend only to the first modality and choose the direction at stimulus onset', 
									'at stimulus onset select the direction that appears in the first modality', 
									'pay attention only to the first modality and select the orientation immediately',
									'focus only on the first modality and choose the displayed orientation at stimulus onset',
									'choose the direction displayed in the first modality at stimulus onset',
									'select the orientation in the first modality immediately', 
									'attend to the first modality and go in the displayed direction immediately', 
									'focus on the first modality and opt for the orientation at stimulus onset', 
									'pay attention to the first modality and select the direction immediately', 
									'immediately respond to the direction in the first modality', 
									'select the orientation that appears in the first modality at stimulus onset', 
									'attend to the first modality and select the displayed direction immediately',
									'go in the direction displayed in the first modality at stimulus onset',
									'attend to the first modality and pick the direction immediately'
									)

train_instruct_dict['RTGoMod2'] = ('focus only on the second modality and choose the stimulus immediately', 
									'attend to the second modality and select the orientation at stimulus onset', 
									'pay attention to the second modality and go in the displayed direction immediately',
									'select the stimulus in the second modality at stimulus onset', 
									'focus on the second modality and select the displayed orientation immediately',
									'attend to the second modality and respond to the displayed direction immediately',
									'choose the stimulus in the second modality immediately',
									'at stimulus onset choose the orientation in the second modality',
									'focus only on the second modality and choose the orientation at stimulus onset',
									'pay attention to the second modality and pick the stimulus immediately',
									'attend to the second modality and respond to the displayed direction at stimulus onset',
									'pay attention to the second modality and opt for the displayed orientation at stimulus onset',
									'respond to the direction in the second modality at stimulus onset',
									'go in the direction displayed in the second modality immediately',
									'opt for the stimulus in the second modality immediately')

train_instruct_dict['AntiRTGoMod1'] = ('focus on the first modality and select the opposite of the stimulus immediately',
									'opt for the reverse of the orientation in the first modality at stimulus onset',
									'choose the opposite of the direction in the first modality immediately',
									'attend only to the first modality and select the reverse of the stimulus immediately',
									'pay attention only to the first modality and opt for the opposite of the orientation at stimulus onset',
									'attend only to the first modality and respond with the opposite direction at stimulus onset',
									'go in the opposite of the orientation displayed in the first modality at stimulus onset',
									'focus on the first modality and respond in the reverse of the stimulus immediately',
									'pay attention to the first modality and go in the reverse of the stimulus immediately',
									'focus on the first modality and choose the reverse of the orientation at stimulus onset',
									'attend only to the first modality and opt for the reverse direction at stimulus onset',
									'pay attention to the first modality and respond to the reverse of the stimulus immediately',
									'go in the reverse of the direction in the first modality at stimulus onset', 
									'respond with the opposite of the stimulus in the first modality immediately',
									'attend to the first modality and select the reverse of the orientation at stimulus onset')

train_instruct_dict['AntiRTGoMod2'] = ('focus only on the second modality and choose the opposite of the stimulus immediately',
										'attend only to the second modality and go in the reverse of the direction at stimulus onset',
										'pay attention to the second modality and respond with the reverse of the stimulus immediately',
										'focus only on the second modality and select the opposite of the orientation at stimulus onset',
										'attend to the second modality and opt for the reverse of the displayed direction immediately',
										'pay attention to the second modality and pick the opposite orientation at stimulus onset',
										'go in the reverse of the direction in the second modality at stimulus onset', 
										'opt for the opposite of the stimulus in the second modality immediately',
										'choose the reverse of the orientation in the second modality at stimulus onset',
										'pay attention to the second modality and choose the opposite of the stimulus immediately',
										'attend to the second modality and opt for the reverse direction at stimulus onset', 
										'respond in the opposite of the direction in the second modality at stimulus onset',
										'focus on the second modality and select the reverse of the displayed stimulus immediately',
										'attend to the second modality and go in the opposite direction at stimulus onset',
										'go in the opposite of the stimulus in the second modality immediately')


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

train_instruct_dict['AntiDM'] = ('respond in the direction of minimal strength', 
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

train_instruct_dict['AntiMultiDM'] = ('respond in the direction with lowest combined value between two stimuli',
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

train_instruct_dict['ConAntiDM'] = ('opt for the stimulus presented with least strength if you are confident in your answer otherwise do not respond', 
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

train_instruct_dict['DMMod1'] = ('select the orientation in the first modality that is strongest', 
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

train_instruct_dict['DMMod2'] = ('choose the direction in the second modality that appears strongest', 
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

train_instruct_dict['AntiDMMod1'] = ('attend to the first modality and choose the weakest direction', 
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

train_instruct_dict['AntiDMMod2'] = ('attend to the second modality and choose the orientation that appears weakest', 
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


train_instruct_dict['Dur1'] =('respond to the first direction if it lasts for longer than the final direction otherwise do not respond',
							'if the first stimulus is presented for a greater period of time than the latter stimulus then respond to the first stimulus otherwise do not respond', 
							'select the initial orientation if it has a duration which is greater than the second orientation otherwise do not respond', 
							'opt for the initial stimulus if it is displayed for more time than the final stimulus otherwise do not respond', 
							'choose the first orientation if it has a greater duration than the latter orientation otherwise do not respond',
							'if the first stimulus is presented for longer than the second stimulus respond to the first stimulus otherwise do not respond', 
							'if the initial orientation is displayed for more time than the final orientation then choose the first orientation otherwise do not respond', 
							'go in the first direction if it appears for a greater amount of time than the latter stimulus otherwise do not respond', 
							'select the initial direction if it appears for longer than the second direction otherwise do not respond', 
							'if the first direction is displayed for more time than the final stimulus respond to the first direction otherwise do not respond', 
							'choose the initial orientation if it has a greater duration than the latter orientation otherwise do not respond', 
							'if the duration of the first orientation is greater than the second orientation respond to the first orientation otherwise do not respond', 
							'if the duration of the initial stimulus is greater than the duration of the final stimulus respond to the first stimulus otherwise do not respond', 
							'pick the first direction if it lasts for more time than the latter direction otherwise do not respond',
							'pick the initial stimulus if it lasts for a greater period of time than the second stimulus otherwise do not respond'
							)

train_instruct_dict['Dur2'] = ('select the second direction if it lasts for longer than the first direction otherwise do not respond', 
								'if the duration of the final orientation is greater than the duration of the first orientation respond to the first orientation otherwise do not respond', 
								'if the duration of the later stimulus is greater than the duration of the first stimulus respond to the first stimulus otherwise do not respond', 
								'pick the second direction if it has a duration that is greater than the first direction otherwise do not respond', 
								'choose the final direction if it is displayed for more time than the first direction otherwise do not respond', 
								'opt for the latter orientation if it lasts for longer than the first orientation otherwise do not respond', 
								'if the final stimulus is presented for a longer time period than the first stimulus then respond to the final stimulus otherwise do not respond', 
								'if the second direction is displayed for more time than the first direction respond to the second direction otherwise do not respond', 
								'respond to the final orientation if is displayed for more time than the first orientation otherwise do not respond',
								'select the second stimulus if it is longer than the first stimulus otherwise do not respond', 
								'if the duration of the latter direction is greater than the duration of the first direction respond to the latter direction otherwise do not respond', 
								'choose the second direction if it is presented for longer than the first direction otherwise do not respond', 
								'if the latter stimulus appears for more time than the first stimulus than select the second stimulus otherwise do not respond',
								'if the duration of the latter stimulus is greater than the duration of the first stimulus than respond to the latter stimulus otherwise do not respond',
								'choose the second direction if it lasts for a greater period of time than the first direction otherwise do not respond'
								)

train_instruct_dict['MultiDur1'] = ('respond to the first direction if it has longer duration averaged over both modalities than the second direction otherwise do not respond', 
                                'if the first direction lasts for longer when combined over both modalities than the final direction then go in that direction otherwise do not respond', 
                                'opt for the initial stimulus if it has a greater duration averaged both modalities than the latter stimulus otherwise do not respond', 
                                'if the initial orientation appears for longer than the second orientation when considered across both modalities than select that orientation otherwise do not respond', 
                                'pick the initial direction if it last for a longer period of time than the final direction in both modalities otherwise do not respond', 
                                'if the first direction appears for a longer period of time then the latter direction over both modalities then select that direction otherwise do  not respond', 
                                'choose the initial stimulus if it has a greater duration than the final stimulus when considered over both modalities otherwise do not respond', 
                                'respond to the initial direction if it is displayed for a greater period of time than the second direction averaged over both modalities otherwise do not respond', 
                                'if the initial orientation lasts for longer than the second orientation when considered over both modalities then select that direction otherwise do not respond', 
                                'choose the first displayed direction if it has a duration which is longer than the final direction combined over both modalities otherwise do not respond', 
                                'if the initial stimulus appears for a longer period of time than the second stimulus averaged across modalities then choose that direction otherwise do not respond', 
                                'select the first stimulus if it has a duration which is longer than the latter direction combined across both modalities otherwise do not respond',
                                'pick the initial orientation if if lasts for longer than the second direction averaged over modalities otherwise do not respond',
                                'if the first direction lasts for a greater period of time than the final direction in both modalities then respond to the second direction otherwise do not respond',
                                'select the initial orientation if it has a greater averaged duration over both modalities than the later orientation otherwise do not respond'
                                )

train_instruct_dict['MultiDur2'] = ('respond to the second direction if it lasts for a longer period of time over both modalities than the first direction otherwise do not respond',
                                    'choose the final direction if it has a greater duration than the first direction averaged across modalities otherwise do not respond', 
									'if the latter orientation appears for a longer period of time when combined over both modalities than the initial direction choose the latter direction otherwise do not respond',
									'if the second stimulus has a greater duration than the first stimulus combined over modalities than select the second stimulus otherwise do not respond', 
									'choose the latter direction if it is longer than the first direction averaged over both modalities otherwise do not respond', 
									'if the duration of the final stimulus is longer than the initial stimulus combined across modalities then choose the final stimulus otherwise do not respond', 
									'select the final orientation if it appears for longer on averaged over modalities than the first orientation otherwise do not respond', 
									'opt for the second direction if it lasts for a greater period of time than the first direction averaged over both modalities otherwise do not respond', 
									'if the time period of the final orientation lasts for longer than the first orientation when averaged over both modalities then respond to the final orientation otherwise do not respond', 
									'pick the second direction if it appears for a greater duration than the first direction when considered over both modalities otherwise do not respond', 
									'if the duration of the latter direction averaged across both modalities is greater than the first direction then choose the latter direction otherwise do not respond', 
									'if the final stimulus lasts for longer than the first stimulus when combined over both modalities then select the final stimulus otherwise do not respond', 
									'respond to the final direction if lasts for a longer period of time than the first stimulus averaged over both modalities otherwise do not respond',
									'if the second stimulus has a duration which lasts for longer than the initial stimulus when combined across both modalities then select the first stimulus otherwise do not respond',
									'select the latter direction if it is displayed for a greater period of time than the initial direction when averaged over both modalities otherwise do not respond'
									)

train_instruct_dict['AntiDur1'] = ('respond to the first direction if it has a shorter duration than the second direction otherwise do not respond',
                                  'choose the initial orientation if it lasts for a shorter period of time than the final orientation otherwise do not respond',
                                  'if the first stimulus has a shorter duration than the latter stimulus then select the first stimulus otherwise do not respond',
                                  'if the initial direction is lasts for less time than the final direction choose the initial direction otherwise do not respond',
                                  'opt for the first orientation if it lasts for less time than the second orientation otherwise do not respond',
                                  'select the initial stimulus if it has a shorter duration than the final stimulus otherwise do not respond', 
                                  'pick the first direction if it lasts for less time than the final direction otherwise do not respond',
                                  'go in first direction if it has a shorter duration than the latter direction otherwise do not respond',
                                  'respond to the first orientation if it is shorter than the final orientation otherwise do not respond', 
                                  'if the initial direction is displayed for less time than the latter direction select the initial direction otherwise do not respond',
                                  'if the first orientation appears for less time than the second direction choose the first orientation otherwise do not respond',
                                  'if the initial stimulus is shorter than the final stimulus respond to the initial stimulus otherwise do not respond', 
                                  'respond to the initial orientation if it is shorter than the second orientation otherwise do not respond', 
                                  'select the first direction if it has a shorter duration than the final direction otherwise do not respond', 
                                  'pick the initial stimulus if it is displayed for less time than the second stimulus otherwise do not respond'
                                  )

train_instruct_dict['AntiDur2'] = ('respond to the final direction if it has a shorter duration than the first direction otherwise do not respond',
                                  'choose the latter stimulus if it is displayed for less time than the first stimulus otherwise do not respond',
                                  'if the second orientation appears for less time than the initial orientation select the second orientation otherwise do not respond',
                                  'if the latter direction is shorter than the initial direction pick the latter direction otherwise do not respond', 
                                  'opt for the final direction if it has a shorter duration than the first direction otherwise do not respond',
                                  'respond to the second stimulus if it is displayed for less time than the first stimulus otherwise do not respond', 
                                  'if the final stimulus is shorter than the initial stimulus choose the final stimulus otherwise do not respond', 
                                  'if the second direction has a shorter duration than the initial direction select the second direction otherwise do not respond',
                                  'if the second orientation is shorter than the first orientation pick the second orientation otherwise do not respond',
                                  'select the latter stimulus if it has a shorter duration than the initial stimulus otherwise do not respond', 
                                  'opt for the final orientation if it is displayed for less time than the initial orientation otherwise do not respond',
                                  'if the latter direction appears for less time than the first direction respond to the latter direction otherwise do not respond',
                                  'choose the final orientation if it is shorter than the initial orientation otherwise do not respond', 
                                  'select the second stimulus if it has a shorter duration than the first stimulus otherwise do not respond', 
                                  'respond to the latter direction if it is shorter than the first direction otherwise do not respond'
                                  )

train_instruct_dict['AntiMultiDur1'] = ('respond to the initial direction if it is displayed for a shorter amount of time than the second direction averaged over modalities otherwise do not respond',
                                  'if the first stimulus is shorter than the final stimulus combined over both modalities then choose the first stimulus otherwise do not respond', 
                                  'if the initial orientation has a shorter duration than the latter orientation considered over both modalities then respond to the initial orientation otherwise do not respond',
                                  'choose the first stimulus if it is shorter than the final stimulus averaged over both modalities otherwise do not respond',
                                  'select the initial direction if it is displayed for less time than the latter direction combined over both modalities otherwise do not respond',
                                  'if the first direction is displayed for less time than the final direction averaged over both modalities then opt for the first direction otherwise do not respond',
                                  'if the initial stimulus has a shorter duration than the second stimulus considered over both modalities then pick the initial stimulus otherwise do not respond',
                                  'opt for the first orientation if it is shorter than the second orientation averaged over both modalities otherwise do not respond', 
                                  'pick the first stimulus if it appears for less time than the final stimulus combined over both modalities otherwise do not respond', 
                                  'respond to the initial orientation if it is displayed for less time averaged over both modalities than the second orientation otherwise do not respond',
                                  'if the first direction is shorter on averaged over both modalities than the final direction then respond to the first direction otherwise do not respond',
                                  'if the initial stimulus has a shorter duration when combined over both modalities than the latter stimulus pick the initial stimulus otherwise do not respond',
                                  'pick the first stimulus if it appears for less time considered over both modalities than the second stimulus otherwise do not respond', 
                                  'if the initial orientation appears for less time averaged over both modalities than the latter stimulus choose the initial orientation otherwise do not respond', 
                                  'go in the first direction if it is shorter combined over both modalities than the second direction otherwise do not respond')


train_instruct_dict['AntiMultiDur2'] = ('select the final direction if it is shorter than the first direction averaged over both modalities otherwise do not respond', 
                                    'respond to the second orientation if is displayed for less time than the initial orientation considered across both modalities otherwise do not respond', 
                                    'if the latter stimulus is displayed for less time combined over both modalities than the initial stimulus then select the latter stimulus otherwise do not respond',
                                    'if the final stimulus is shorter than the first stimulus averaged over both modalities than select the final stimulus otherwise do not respond',
                                    'choose the latter orientation if it has a shorter duration averaged over both modalities than the initial orientation otherwise do not respond',
                                    'if the second direction has a shorter duration than the first direction considered across both modalities than select the second direction otherwise do not respond',
                                    'if the final orientation is shorter than the initial orientation combined across both modalities then choose the final orientation otherwise do not respond', 
                                    'opt for the latter stimulus it it is presented for less time than the initial stimulus considered across both modalities otherwise do not respond',
                                    'pick the final direction if it has a shorter duration averaged over both modalities than the first direction otherwise do not respond', 
                                    'go in the second direction if it is shorter when combined over both modalities than the first direction otherwise do not respond',
                                    'respond to the latter direction if it is displayed for less time than the initial direction combined over both modalities otherwise do not respond',
                                    'if the second stimulus is shorter than the first stimulus combined over both modalities then opt for the second stimulus otherwise do not respond',
                                    'if the final direction is displayed for less time combined over both modalities than the first direction then choose the final direction otherwise do not respond',
                                    'select the second orientation if it has a shorter duration considered across both modalities than the initial orientation otherwise do not respond',
                                    'pick the final direction if it is shorter than the first direction averaged over both modalities otherwise do not respond')



train_instruct_dict['Dur1Mod1'] = ('attend to the first modality and choose the first direction if has a longer duration than the second direction otherwise do not respond',
									'attend only to the first modality and select the initial stimulus if it lasts for a longer period of time than the latter stimulus otherwise do not respond', 
									'if the first orientation lasts for longer than the final orientation in the first modality then respond to the first orientation otherwise do not respond', 
									'focus only on the first modality and opt for the initial direction if it has a greater duration than the second direction otherwise do not respond', 
									'pay attention only to the first modality and if the first stimulus lasts for more time than the final stimulus then respond to the first stimulus otherwise do not respond', 
									'select the first direction if it is presented for a longer duration than the latter direction in the first modality otherwise do not respond', 
									'if the initial orientation is displayed for more time than the second orientation in the first modality then respond to the initial orientation otherwise do not respond', 
									'focus only on the first modality and opt for the first stimulus if it is presented for longer than the final stimulus otherwise do not respond', 
									'focus on the first modality and pick the first orientation if it appears for a greater period of time than the latter orientation otherwise do not respond', 
									'go in the first direction if the is has greater duration than the second stimulus in the first modality otherwise do not respond', 
									'attend only to the first modality and choose the first direction if is appears for more time than the final direction otherwise do not respond', 
									'pay attention to the first modality and respond to the initial orientation if it has a duration that is greater than the latter stimulus otherwise do not respond', 
									'pay attention only to the first modality and select the initial stimulus if is lasts for longer than the second stimulus otherwise do not respond',
									'attend only to the first modality and pick the initial orientation if it has a longer duration than the final orientation otherwise do not respond',
									'focus on the first modality and opt for the first direction if it lasts for longer than the second direction otherwise do not respond')

train_instruct_dict['Dur1Mod2'] = ('attend only to the second modality and select the initial direction if it lasts for longer than the final direction otherwise do not respond', 
									'if the first orientation is displayed for more time than the final direction in the second modality then respond to the first orientation presented there otherwise do not respond',
									'attend only to the second modality and pick the first stimulus if it lasts for a greater period of time than the latter direction otherwise do not respond', 
									'pay attention only to the second modality and opt for the initial direction if it has a longer duration than the final direction otherwise do not respond',
									'focus on the second modality and select the first stimulus if it appears for a greater amount of time than the second stimulus otherwise to not respond', 
									'focus on the second modality and choose the initial orientation if it is presented for longer than the final orientation otherwise do not respond', 
									'go in the first direction if it has a greater duration than the latter direction in the second modality otherwise do not respond', 
									'select the initial stimulus if it has a greater duration than the final stimulus in the second modality otherwise do not respond',
									'if the duration of the first stimulus is longer than the duration of the latter stimulus in the second modality respond to the first stimulus otherwise do not respond', 
									'if the first direction lasts for a greater period of time than the final direction in the second modality then select the first direction otherwise do not respond', 
									'pay attention to the second modality and pick the first orientation if it appears for a greater period of time than the second orientation otherwise do not respond',
									'pay attention only to the second modality and select the initial direction if it lasts for longer than the second direction otherwise do not respond', 
									'attend to the second modality and if the duration of the initial stimulus is greater than that of the final stimulus then respond to the initial direction otherwise do not respond', 
									'focus on the second modality and choose the first orientation if it lasts for more time than the second orientation otherwise do not respond',
									'attend only to the second modality and opt for the initial direction if it lasts for longer than the latter direction otherwise do not respond')

train_instruct_dict['Dur2Mod1'] = ('attend only to the first modality and select the second direction if it lasts for a greater period of time than the first direction otherwise do not respond', 
									'pay attention only to the first modality and if the final orientation has a longer duration than the initial direction respond to the first orientation otherwise do not respond', 
									'focus only on the first modality and choose the latter direction if it lasts for more time than the first direction otherwise do not respond', 
									'select the second direction if it is displayed for a greater period of time than the first direction in the first modality otherwise do not respond', 
									'focus only on the first modality and pick the final orientation if it has a longer duration than the initial direction otherwise do not respond', 
									'pay attention only to the first modality and if the second stimulus is displayed for more time than the first stimulus respond to the second stimulus otherwise do not respond', 
									'if the duration of the final stimulus is longer than the first stimulus in the first modality then respond to the final stimulus otherwise do not respond', 
									'if the latter direction lasts for a greater period of time than the initial direction in the first modality then select the latter direction otherwise do not respond', 
									'attend only to the first modality and opt for the second stimulus if it appears for a greater period of time than the initial stimulus otherwise do not respond', 
									'choose the second orientation if it has a greater duration than the initial orientation in the first modality otherwise do not respond', 
									'focus only on the first modality and choose the second direction if it is longer than the initial direction otherwise do not respond', 
									'attend to the first modality and opt for the final orientation if it lasts for longer than the first orientation otherwise do not respond',
									'pay attention only to the first modality and choose the latter stimulus if it is displayed for a longer period of time than the first stimulus otherwise do not respond',
									'respond to the second stimulus if it appears for a longer period of time than the first stimulus in the first modality otherwise do not respond',
									'attend to the first modality and respond to the latter orientation if it lasts for a greater period of time than the first orientation otherwise do not respond')

train_instruct_dict['Dur2Mod2'] = ('attend only to the second modality and respond to the final direction if it has a longer duration than the first direction otherwise do not respond', 
									'focus only on the second modality and choose the latter stimulus if it appears for a greater period of time than the first direction otherwise do not respond', 
									'if the second orientation appears for a greater period of time than the initial orientation in the second modality then select the second orientation otherwise do not respond',
									'if the duration of the second stimulus is longer than that of the first stimulus in the second modality then respond to the second stimulus otherwise do not respond', 
									'focus on the second modality and choose the final orientation if it lasts for longer than the first orientation otherwise do not respond', 
									'attend to the second modality and select the latter stimulus if it is displayed for a greater period of time than the first stimulus otherwise do not respond', 
									'choose the second direction if it has a greater duration than the first direction in the second modality otherwise do not respond', 
									'respond to the second orientation if it lasts for longer than the initial orientation in the second modality otherwise do not respond', 
									'if the duration of the final direction is longer than the duration of the first direction in the second modality then respond to the final direction otherwise do not respond',
									'pay attention only to the second modality and if the final stimulus lasts for more time than the first stimulus then respond to the second stimulus otherwise do not respond',
									'pay attention to the second modality and pick the latter direction if it is displayed for more time than the first direction otherwise do not respond', 
									'choose the second orientation if it has a longer duration than the first orientation in the second modality otherwise do not respond', 
									'attend only to the second modality and opt for the final stimulus if it lasts for a greater period of time than the first stimulus otherwise do not respond', 
									'attend only to the second modality and select the latter orientation if it has a longer duration than the first orientation otherwise do not respond',
									'pick the second direction if it appears for more time than the first direction in the second modality otherwise do not respond'
									)

train_instruct_dict['COMP1'] = ('if the first stimulus is greater than the second stimulus respond to the first stimulus otherwise do not respond', 
                                'if the first stimulus has higher intensity than the final go in the first direction otherwise do not respond', 
                                'go in the direction of the first stimulus if it is stronger than the second stimulus otherwise do not respond', 
                                'when the initial stimulus has higher value than the final stimulus respond in the initial direction otherwise do not respond', 
                                'if the initial stimulus has greater strength than the latter stimulus respond in the first direction otherwise do not respond',   
                                'respond to the initial stimulus if it is stronger than the last stimulus otherwise do not respond', 
                                'choose the initial stimulus when it is presented with higher intensity relative to the second stimulus otherwise do not respond', 
                                'if the initial stimulus has higher value than the second respond to the first direction otherwise do not respond', 
                                'when the first stimulus is presented with the higher intensity than the latter select the first direction otherwise do not respond', 
                                'choose the first stimulus if it is presented with the greater intensity than the second otherwise do not respond',
                                'select the initial stimulus if is stronger than the final stimulus otherwise do not respond',
                                'if the first stimulus has larger value than than the latter stimulus select the first direction otherwise do not respond',
                                'respond to the first direction if it is more intense than the last stimulus otherwise do not respond',
                                'if the first stimulus is more intense than the second stimulus choose the first direction otherwise do not respond',
                                'when the initial stimulus is presented with greater strength than the second stimulus respond to the first direction otherwise do not respond')

train_instruct_dict['COMP2'] = ('respond in the direction of the second stimulus if it has greater intensity than the first otherwise do not respond', 
                                'if the final stimulus is presented with higher value than the initial stimulus go in the final direction otherwise do not respond', 
                                'when the final stimulus has greater strength than the first select the final orientation otherwise do not respond', 
                                'respond in the direction of the second stimulus if it has higher intensity than the initial stimulus otherwise do not respond', 
                                'if the final stimulus has higher value than the first stimulus choose the final orientation otherwise do not respond', 
                                'select the latter stimulus direction if it is presented with greater strength than the first otherwise do not respond', 
                                'when the last stimulus is presented with more intensity than the first respond in the second direction otherwise do not respond', 
                                'choose the final stimulus if it has the greater value than the first otherwise do not respond', 
                                'if the final stimulus is presented with the higher intensity than the initial stimulus respond to the second stimulus otherwise do not respond', 
                                'select the final stimulus if it is presented with the higher intensity than the initial stimulus otherwise do not respond',
                                'when the second stimulus is more intense than the initial stimulus respond to the second stimulus otherwise do not respond',
                                'if the second stimulus is presented with greater strength than the first respond to the second stimulus otherwise do not respond',
                                'choose the latter stimulus if it is more intense than the the first stimulus otherwise do not respond',
                                'when the final stimulus is more intense than the first respond to the final stimulus otherwise do not respond',
                                'pick the second stimulus if it has more intensity than the first otherwise do not respond')

train_instruct_dict['MultiCOMP1'] = ('respond to the first direction when it has greater strength on average than the second stimulus otherwise do not respond',
                                        'respond if the first direction has higher combined intensity over two modalities than the final stimulus otherwise do not respond', 
                                        'if the joint intensity of the first directions is higher than the second then respond to the first direction otherwise do not respond', 
                                        'choose the initial direction if the integrated strength across modalities is greater than the final direction otherwise do not respond', 
                                        'go in the initial direction if it has greater value than the final direction when combined over both modalities otherwise do not respond', 
                                        'select the first direction when it has greater integrated strength across modalities than the second direction otherwise do not respond', 
                                        'if the initial direction has higher overall value over modalities than the final direction then choose the initial direction otherwise do not respond', 
                                        'if the combined intensity of the first direction is higher than the latter direction then respond to the first direction otherwise do not respond', 
                                        'choose the initial direction when the joint strength over both modalities is greater than the final direction otherwise do not respond', 
                                        'pick the first direction if the average value is larger than the second direction for both modalities otherwise do not respond', 
                                        'select the initial direction if it displays a higher overall intensity over modalities than the final direction otherwise do not respond',
                                        'if the intensity of the first stimulus integrated over modalities is greater than the second respond in the first direction otherwise do not respond',
                                        'select the first stimulus if it has higher joint strength over modalities than the final otherwise do not respond',
                                        'if the first direction is presented with higher intensity averaged over modalities than the second direction then respond to the first direction otherwise do not respond',
                                        'pick the initial stimulus if it has greater joint intensity over both modalities than the latter otherwise do not respond')

train_instruct_dict['MultiCOMP2'] = ('respond in the second direction if it has larger overall strength over modalities than the first direction otherwise do not respond', 
                                        'if the final direction has greater combined value than the initial direction then respond in the final direction otherwise do not respond', 
                                        'respond in the latter direction when it has higher integrated value over modalities than the first direction otherwise do not respond', 
                                        'select the second direction if it has greater strength than the first direction over both modalities otherwise do not respond', 
                                        'choose the final direction if it has a joint intensity higher than the initial direction over modalities otherwise do not respond', 
                                        'if the final direction is presented with greater overall strength across modalities than the initial direction choose the final direction otherwise do not respond', 
                                        'respond to the latter direction when it has larger combined value over modalities than the first direction otherwise do not respond', 
                                        'select the final direction if the strength integrated over modalities is greater than the first direction otherwise do not respond', 
                                        'if the final direction is represented with higher joint intensity over both modalities than the initial direction then respond to the final direction otherwise do not respond', 
                                        'choose the second direction when its combined strength over modalities is higher than the first direction otherwise do not respond',
                                        'if the second stimulus is presented with higher joint intensity over modalities than the first then select the second stimulus otherwise do not respond',
                                        'pick the final direction if it has larger strength combined over modalities than the initial direction otherwise do not respond',
                                        'respond to the second stimulus if it is presented with greater intensity averaged over modalities than the first stimulus otherwise do not respond',
                                        'choose the latter direction if it has stronger presentation averaged over modalities than the first direction otherwise do not respond',
                                        'if the final stimulus has higher overall value for both modalities than the initial direction then select the final stimulus otherwise do not respond')

train_instruct_dict['AntiCOMP1'] = ('respond to the first direction if it is weaker than the final direction otherwise do not respond', 
                                    'if the initial stimulus is presented with less intensity than the latter stimulus then select the initial stimulus otherwise do not respond',
									'choose the initial orientation if it has lower intensity than the second orientation otherwise do not respond',
									'if the first direction is displayed less strength than the final direction then select the first direction otherwise do not respond', 
									'opt for the initial orientation if is weaker than the latter orientation otherwise do not respond', 
									'pick the initial stimulus if it is displayed with lower strength than the second stimulus otherwise do not respond',
									'if the first direction is less intense than the final direction then choose the first direction otherwise do not respond', 
									'select the initial orientation if it is presented with lower strength than the latter orientation otherwise do not respond', 
									'if the first stimulus is less intense than the second stimulus respond to the first stimulus otherwise do not respond', 
									'go in the direction of the first stimulus if it has less intensity than the final stimulus otherwise do not respond', 
									'if the initial direction has less strength than the second direction then select the initial direction otherwise do not respond', 
									'if the first stimulus is presented with less strength than the final direction then choose the first stimulus otherwise do not respond', 
									'respond to the initial orientation if it is weaker than the second orientation otherwise do not respond', 
									'select the first stimulus if it is presented with less intensity than the second stimulus otherwise do not respond',
									'choose the first direction if is has less strength than the latter direction otherwise do not respond'
									)

train_instruct_dict['AntiCOMP2'] = ('respond to the final direction if it is weaker than the initial direction otherwise do not respond', 
									'if the second orientation is presented with less strength than the first direction then select the second orientation otherwise do not respond', 
									'if the latter stimulus is less intense than the first stimulus then respond to the latter stimulus otherwise do not respond', 
									'choose the second orientation if it is displayed with lower intensity than the first orientation otherwise do not respond', 
									'select the final stimulus if it is weaker than the first direction otherwise do not respond', 
									'opt for the second direction if it has less strength than the initial direction otherwise do not respond', 
									'if the final orientation is presented with less intensity than the first orientation then select the final orientation otherwise do not respond', 
									'if the second stimulus has lower intensity than the first stimulus respond to the second stimulus otherwise do not respond', 
									'go in the final direction if it has less strength than the initial direction otherwise do not respond', 
									'if the second stimulus has lower intensity than the initial stimulus respond to the second stimulus otherwise do not respond', 
									'if the latter direction is less intense than the initial direction then select the latter direction otherwise do not respond', 
									'pick the second orientation if it is weaker than the initial orientation otherwise do not respond', 
									'respond to the final stimulus if it is weaker than the first stimulus otherwise do not respond',
									'choose the second direction if it is presented with lower intensity than the initial direction otherwise do not respond',
									'if the final stimulus is weaker than the initial stimulus respond to the final stimulus otherwise do not respond'
									)

train_instruct_dict['AntiMultiCOMP1'] = ('respond to the first direction if it is weaker averaged over both modalities than the final direction otherwise do not respond',
										'if the first orientation has less strength over both modalities than the second orientation then select the first orientation otherwise do not respond', 
										'if the first stimulus is less intense than the latter stimulus when averaged over both modalities then choose the stimulus stimulus otherwise do not respond', 
										'pick the first direction if it has lower intensity than the second direction combined over both modalities otherwise do not respond', 
										'select the initial direction if it is weaker than the final direction averaged over both modalities otherwise do not respond',
										'opt for the initial orientation if it has lower overall strength than the latter orientation considered over both modalities otherwise do not respond', 
										'if the initial stimulus is presented with less intensity than the second stimulus averaged over both modalities then select the initial stimulus otherwise do not respond', 
										'if the first direction is weaker averaged over both modalities than the final direction select the first direction otherwise do not respond', 
										'if the initial orientation has less intensity combined over both modalities than the latter orientation respond to the initial orientation otherwise do not respond',
										'select the first direction if it has less overall strength over both modalities than the second direction otherwises do not respond',
										'choose the initial orientation if it is presented with lower overall intensity over both modalities than the final orientation otherwise do not respond', 
										'if the first orientation has lower combined intensity over both modalities than the second orientation then select the first orientation otherwise do not respond', 
										'respond to the initial stimulus if it is presented with less combined strength over both modalities than the latter stimulus otherwise do not respond',
										'if the first direction is presented lower intensity averaged over both modalities than the second direction then select the first direction otherwise do not respond',
										'pick the initial stimulus if it is weaker averaged over both modalities than the final stimulus otherwise do not respond')

train_instruct_dict['AntiMultiCOMP2'] = ('respond to the second direction if it is weaker averaged over both modalities than the first direction otherwise do not respond', 
										'if the final orientation is presented with less intensity combined over both modalities than the first direction select the final orientation otherwise do not respond', 
										'if the second stimulus has lower strength averaged over both modalities than the initial stimulus then select the second stimulus otherwise do not respond',
										'opt for the final direction if is less intense combined over both modalities than the initial direction otherwise do not respond', 
										'pick the second orientation if it has lower overall intensity over both modalities than the first orientation otherwise do not respond', 
										'if the final stimulus is weaker averaged over both modalities than the initial stimulus then select the final stimulus otherwise do not respond', 
										'if the latter direction is displayed with less intensity combined over both modalities than the first direction then select the latter direction otherwise do not respond', 
										'if the second orientation has less strength averaged over both modalities than the initial orientation then select the second orientation otherwise do not respond', 
										'choose the final stimulus if it has lower overall strength over both modalities than the first stimulus otherwise do not respond', 
										'select the latter direction if has less intensity when combined over both modalities than the initial direction otherwise do not respond', 
										'if the second orientation is weaker than the first orientation averaged over both modalities then choose the second orientation otherwise do not respond', 
										'opt for the final stimulus if it has lower intensity averaged over both modalities than the first stimulus otherwise do not respond',
										'if the second direction is presented with weaker intensity combined over both modalities than the first stimulus select the second direction otherwise do not respond', 
										'respond to the final stimulus if it is weaker when combined over both modalities than the first stimulus otherwise do not respond',
										'if the latter orientation is displayed with less intensity averaged over both modalities than the first orientation then choose the latter orientation otherwise do not respond')

train_instruct_dict['COMP1Mod1'] = ('attend only to the first modality and respond to the first direction if is stronger than the second stimulus otherwise do not respond', 
                                    'if the first stimulus has greater strength than the final stimulus in the first modality select the first direction otherwise do not respond', 
                                    'pay attention only to the first modality and select the first orientation if it is presented with greater intensity than the latter orientation otherwise do not respond', 
                                    'focus only on the first modality and choose the initial stimulus if it has more intensity than the second stimulus otherwise do not respond', 
                                    'if the initial direction is stronger than the latter direction in the first modality then respond to the initial direction otherwise do not respond', 
                                    'choose the first direction if it is presented with higher intensity than the final direction in the first modality otherwise do not respond', 
                                    'if the first stimulus has more intensity than the latter stimulus in the first modality then choose the first stimulus otherwise do not respond', 
                                    'attend only to the first modality and select the initial direction if if has more strength than the final stimulus otherwise do not respond', 
                                    'if the first orientation is presented with higher intensity than the second orientation in the first modality then respond to the first orientation otherwise do not respond', 
                                    'pick the first direction if it is stronger than the second direction in the first modality otherwise do not respond', 
                                    'respond to the initial direction if it is presented with more strength than the final direction in the first modality otherwise do not respond', 
                                    'opt for the first orientation if it is more intense than the second direction in the first modality otherwise do not respond', 
                                    'focus only on the first modality and select the initial stimulus if it has greater strength than the final direction otherwise do not respond', 
                                    'if the first orientation has higher intensity than the latter orientation in the first modality then respond to the first orientation otherwise do not respond', 
                                    'pay attention only to the first modality and opt for the first direction if it has more strength than the second direction otherwise do not respond')

train_instruct_dict['COMP1Mod2'] = ('focus only on the second modality and select the first direction if it is stronger than the second direction otherwise do not respond', 
                                    'if the initial orientation has higher intensity than the final orientation in the second modality then select the initial orientation otherwise do not respond',
                                    'pay attention only to the second modality and choose the initial stimulus if it has greater strength than the final stimulus otherwise do not respond', 
                                    'select the first direction if it has more intensity than the latter direction in the second modality otherwise do not respond', 
                                    'if the first stimulus is stronger than the final stimulus in the second modality then select the first stimulus otherwise do not respond', 
                                    'attend only to the second modality and if the first stimulus is greater than the second stimulus respond to the first stimulus otherwise do not respond', 
                                    'choose the first direction if it is presented with greater strength than the final direction in the second modality otherwise do not respond', 
                                    'if the initial direction has greater strength than the latter direction in the second modality then respond to the initial direction otherwise do not respond', 
                                    'focus only on the second modality and opt for the first orientation if it is presented with more intensity than the second orientation otherwise do not respond', 
                                    'select the initial direction if it is displayed with greater strength than the final direction in the second modality otherwise do not respond', 
                                    'pick the first stimulus if it is stronger than the latter stimulus in the second modality otherwise do not respond', 
                                    'attend only to the second modality and select the first orientation if it has greater strength than the second orientation otherwise do not respond', 
                                    'choose the first direction if it has higher intensity than the final direction in the second modality otherwise do not respond', 
                                    'if the initial stimulus is more intense than the last stimulus in the second modality then respond to the initial stimulus otherwise do not respond',
                                    'opt for the first direction if it is stronger than the second direction in the second modality otherwise do not respond')

train_instruct_dict['COMP2Mod1'] = ('attend only to the first modality and select the second direction if it is stronger than the first otherwise do not respond', 
                                    'opt for the final direction if it is presented with greater strength than the initial stimulus in the first modality otherwise do not respond', 
                                    'if the second stimulus is stronger than the first stimulus in the first modality than select the second stimulus otherwise do not respond', 
                                    'focus only on the first modality and choose the second orientation if is presented more intensely than the first direction otherwise do not respond', 
                                    'pick the final direction if it is displayed with higher intensity than the first direction in the first modality otherwise do not respond', 
                                    'pay attention only to the first modality and select the latter stimulus if it is stronger than the first stimulus otherwise do not respond', 
                                    'if the second orientation is greater than the first orientation in the first modality then respond to the second orientation otherwise do not respond', 
                                    'select the second stimulus if it is presented with higher strength than the first stimulus in the first modality otherwise do not respond', 
                                    'if the final orientation is stronger than the first orientation in the first modality than choose the last orientation otherwise do not respond', 
                                    'focus only on the first modality and go in the second direction if it is presented with higher intensity than the first direction otherwise do not respond', 
                                    'respond to the second stimulus if it has greater strength than the first stimulus in the first modality otherwise do not respond', 
                                    'attend only to the first modality and opt for the final direction if it is more intense than the initial direction otherwise do not respond', 
                                    'go in the direction of the latter stimulus if it has greater strength than the initial stimulus in the first modality otherwise do not respond',
                                    'focus only on the first modality and choose the the final direction if it is stronger than the first direction otherwise do not respond',
                                    'select the second orientation if it is greater than the first orientation in the first modality otherwise do not respond')

train_instruct_dict['COMP2Mod2'] = ('focus only on the second modality and select the latter direction if it is stronger than the first direction otherwise do not respond',
                                    'select the final direction if it is greater than the initial direction in the second modality otherwise do not respond', 
                                    'pay attention only to the second modality and pick the final orientation if it has higher intensity than the first orientation otherwise do not respond', 
                                    'if the final stimulus is presented with higher intensity than the first stimulus in the second modality then respond to the final stimulus otherwise do not respond', 
                                    'select the second orientation if it is displayed more intensely than the first orientation in the second modality otherwise do not respond',
                                    'if the final stimulus is greater than the initial stimulus in the second modality than respond to the first stimulus otherwise do not respond', 
                                    'attend only to the second modality and if the final stimulus is stronger than the initial stimulus select the final stimulus otherwise do not respond', 
                                    'pick the second stimulus if it is presented with greater strength than the first stimulus in the second modality otherwise do not respond', 
                                    'choose the final orientation if it is greater than the initial orientation in the second modality otherwise do not respond', 
                                    'focus only on the second modality and pick the latter direction if it has more strength than the initial direction otherwise do not respond', 
                                    'go in the direction of the second stimulus if it has more intensity than the initial stimulus in the second modality otherwise do not respond', 
                                    'select the second direction if it is stronger than the first direction in the second modality otherwise do not respond', 
                                    'choose the final stimulus if it has higher intensity than the initial stimulus in the second modality otherwise do not respond', 
                                    'attend only to the second modality and select the final direction if it has higher intensity than the first direction otherwise do not respond', 
                                    'if the latter stimulus is stronger than the first stimulus in the second modality respond to the latter stimulus otherwise do not respond')

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
                                'when stimuli are presented in different directions respond to the latter direction otherwise do not respond', 
                                'respond in the final direction if stimuli are mismatched otherwise do not respond', 
                                'go in the second direction when stimuli orientations are different otherwise do not respond', 
                                'respond in the final displayed direction if stimuli are mismatched otherwise do not respond', 
                                'if stimuli are mismatched respond in the latter direction otherwise do not respond', 
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
                            'if the initial and second direction are in the same half of the display then respond to the initial direction otherwise do not respond',
                            'select the initial stimulus if the initial and final stimuli are presented on the same half of the display otherwise do not respond',
                            'if the displayed directions are on the same half of the display then respond to the first direction otherwise do not respond',
                            'choose the first direction if both presented stimuli are on the same half of the display otherwise do not respond')

train_instruct_dict['DNMC'] = ('if stimuli are on different halves of display respond in the second direction otherwise do not respond', 
                                'when the stimuli are presented are on opposing sides of the display go in the final direction otherwise do not respond', 
                                'go in the final direction when stimuli are on distinct halves of the display otherwise do not respond', 
                                'if the directions are on different halves select the second direction otherwise do not respond', 
                                'when the stimuli appear in distinct halves choose the latter direction otherwise do not respond', 
                                'select the final direction if stimuli are displayed on opposing sides of display otherwise do not respond', 
                                'choose the latter direction when stimuli appear on different halves otherwise do not respond', 
                                'if the stimuli are displayed on distinct sides then respond in second direction otherwise do not respond', 
                                'select the final orientation if stimuli are on different sides of display otherwise do not respond', 
                                'respond in the latter direction when directions are presented on different halves of the display otherwise do not respond',
                                'if the stimuli are presented on opposing sides of the display then select the second stimulus otherwise do not respond',
                                'if the first and second directions are displayed on opposite halves of the display respond in the final direction otherwise do not respond',
                                'go in the second direction when the stimuli are presented on opposite halves of the display otherwise do not respond',
                                'respond to the final orientation if both are presented on distinct halves otherwise do not respond',
                                'choose the final stimulus if both presented stimuli are on opposite halves of the display otherwise do not respond')


test_instruct_dict = {}

test_instruct_dict['Go'] = ('select the presented direction', 
                                'respond in the direction shown',
                                'respond in the direction displayed', 
                                'respond with the displayed orientation', 
                                'choose the presented direction')

test_instruct_dict['RTGo'] = ('choose the direction shown immediately', 
                                    'select the direction presented immediately', 
                                    'select the displayed orientation as soon as stimulus appears', 
                                    'choose the direction presented immediately', 
                                    'immediately respond to the presented orientation')

test_instruct_dict['AntiGo'] = ('select the opposing direction', 
                                        'choose the opposite of the presented direction', 
                                        'respond with the reverse orientation', 
                                        'go in the opposite of displayed orientation',
                                        'pick the converse of the stimulus orientation')

test_instruct_dict['AntiRTGo'] = ('pick the converse of the stimulus direction immediately', 
                                        'select the reverse of the displayed direction immediately', 
                                        'respond with the reverse orientation as soon as it is shown', 
                                        'choose the opposite of displayed orientation at stimulus onset',
                                        'immediately respond in the opposing direction as is shown')

test_instruct_dict['GoMod1'] = ('choose the orientation in the first modality',
                                'opt for the stimulus that appears in the first modality',
                                'only consider the first modality and respond in to the displayed direction',
                                'go in the direction stimulus in the first modality',
                                'concentrate only on the first modality and choose the direction displayed there')

test_instruct_dict['GoMod2'] =('respond to the stimulus in the second modality',
                              'select the direction that is displayed in the second modality',
                              'only consider the second modality and choose the displayed direction',
                              'opt for the orientation in the second modality',
                              'concentrate only on the first modality and select the direction that appears there')

test_instruct_dict['AntiGoMod1'] =('go in the opposite direction of the stimulus presented in the first modality',
                                    'concentrate only on the first modality and choose the converse direction',
                                    'select the opposite of the direction displayed in the first modality',
                                    'pick the direction opposite of the one displayed in the first modality',
                                    'only consider stimulus in the first modality and respond in the opposite direction')

test_instruct_dict['AntiGoMod2'] =('concentrate only on the second modality and go in the reverse of the displayed orientation',
                                        'opt for the opposite of the direction in the second modality',
                                       'go in the converse of the direction displayed in the second modality',
                                     'only consider the second modality and respond in the reverse direction' ,
                                     'pick the opposite of the orientation in the second modality')

test_instruct_dict['RTGoMod1'] = ('concentrate only on the first modality and select the displayed stimulus immediately',
                                  'choose the direction in the first modality as soon as it appears', 
                                  'immediately choose the orientation in the first modality', 
                                  'focus exclusively on the first modality and respond to the stimulus immediately', 
                                  'select the direction in the first modality at stimulus onset')                            

test_instruct_dict['RTGoMod2'] =('attend exclusively to the second modality and choose the stimulus as soon as it appears', 
                                'as soon as the stimulus appears select the direction in the second modality', 
                                'respond to the stimulus that is shown in the second modality immediately', 
                                'only consider the second modality and select the displayed direction as soon as it appears', 
                                'select the direction shown in the second modality as soon as it appears')

test_instruct_dict['AntiRTGoMod1'] =('attend exclusively to the first modality and respond to the opposite direction of the stimulus as soon as it appears', 
                                      'respond to the opposite of the stimulus in the first modality immediately', 
                                      'only consider stimuli in the first modality and choose the opposite of the displayed direction at stimulus onset', 
                                      'go in the opposite of the orientation displayed in the first modality as soon as it appears', 
                                      'select the reverse of the direction displayed in the first modality immediately')

test_instruct_dict['AntiRTGoMod2'] =('attend exclusively to the second modality and go in the opposite of the direction displayed there at once', 
                                      'concentrate only on the second modality and respond to the reverse of the stimulus immediately', 
                                      'choose the opposite of the orientation in the second modality at stimulus onset', 
                                      'opt for the reverse of direction in the second modality immediately', 
                                      'select the opposite of the stimulus that is displayed in the second modality as soon as it is shown')


test_instruct_dict['DM'] = ('select the stimulus of greatest strength', 
                                'choose the direction with maximal stimulus intensity', 
                                'pick the direction with stimulus of greatest strength', 
                                'select the orientation displayed with greatest intensity', 
                                'go in the direction of the most intense stimulus')

test_instruct_dict['AntiDM'] = ('choose the direction with lowest value', 
                                    'respond in the direction that is presented with least intensity', 
                                    'pick the orientation with the lowest strength', 
                                    'go in the direction with weakest presentation strength',
                                    'select the direction with minimal intensity')

test_instruct_dict['MultiDM'] = ('choose the stimuli with highest intensity averaged over modalities', 
                                    'select the orientation with greatest combined strength over modalities', 
                                    'go in direction with highest joint intensity between stimuli', 
                                    'pick the direction of stimulus with greatest intensity between two modalities',
                                    'select the stimuli with maximal combined strength over both modalities')

test_instruct_dict['AntiMultiDM'] = ('pick the direction with minimal average value over modalities', 
                                            'select the orientation with lowest combined strength', 
                                            'go in the direction of the stimuli with weakest overall value between modalities', 
                                            'choose the direction with lowest presentation strength over modalities',
                                            'select the orientation with weakest intensity over both modalities')

test_instruct_dict['ConDM'] = ('choose the strongest stimulus if you are convinced of your answer otherwise do not respond',
                                'select the stimulus with greatest strength if there is no doubt in your mind otherwise do not respond', 
                                'if you are positive then choose the direction presented with greatest strength otherwise do not respond', 
                                'opt for the strongest direction displayed if you are certain about your answer otherwise do not respond', 
                                'respond to the orientation with greatest strength if you are certain of the answer otherwise do not respond')

test_instruct_dict['ConAntiDM'] = ('select the weakest direction if you are convinced of the answer otherwise do not respond', 
                                	'respond to the stimulus with weakest strength if you are positive otherwise do not respond', 
									'if you are certain of your answer select the weaker of the two directions otherwise do not respond', 
									'if there is no doubt in your mind choose the stimulus with least strength otherwise do not respond', 
									'opt for direction with lowest strength if you are certain about your answer otherwise do not respond')

test_instruct_dict['DMMod1'] = ('concentrate only on the first modality and choose the strongest direction', 
								'focus only on the first modality and opt for the direction displayed with the greatest strength',
								'attend exclusively to stimuli in the first modality and choose the strongest one', 
								'only consider the first modality and choose the strongest direction', 
								'consider only the stimuli in the first modality and opt for the stimulus presented with highest strength')

test_instruct_dict['DMMod2'] = ('concentrate only on the second modality and select the strongest direction', 
								'focus exclusively on the second modality and respond to the orientation presented with greatest strength', 
								'only consider stimuli presented in the second modality and choose the strongest among them', 
								'attend to the second modality and opt for the direction displayed with the most strength',
								'consider only the stimuli in the second modality and select the one with the highest strength')

test_instruct_dict['AntiDMMod1'] = ('concentrate exclusively on the first modality and opt for the weakest direction', 
									'only consider stimuli in the first modality and select the weakest of the displayed directions', 
									'consider only the first modality and choose the direction with least strength', 
									'focus exclusively on the first modality and respond to the weakest direction presented there', 
									'concentrate on only the stimuli in the first modality and respond to the weakest direction')

test_instruct_dict['AntiDMMod2'] = ('attend exclusively to the second modality and opt for the direction with least strength', 
										'concentrate only on the second modality select the orientation that appears weakest', 
										'only consider the second modality and respond to the first presented with least strength', 
										'consider only the stimuli presented in the second modality and choose the weakest one that appears there', 
										'focus exclusively on the second modality and opt for the weakest direction')

test_instruct_dict['Dur1'] = ('select the initial stimulus if it is presented for a greater span of time than the second direction otherwise do not respond',
								'if the first orientation is displayed for a greater length of time than the second direction respond to the first orientation otherwise do not respond', 
								'choose the initial direction if is displayed for a longer span of time than the second direction otherwise do not respond',
								'opt for the first stimulus if it lasts for a greater length of time than the second stimulus otherwise do not respond', 
								'if the initial orientation lasts for a larger span of time than the second orientation then select the initial orientation otherwise do not respond')

test_instruct_dict['Dur2'] = ('respond to the final direction if the stimulus is presented for a greater span of time than the first stimulus otherwise do not respond',
								'if the second direction lasts for a greater length of time than the first direction then select the second direction otherwise do not respond',
								'select the second orientation if it appears for a longer span of time than the first direction otherwise do not respond', 
								'opt for the final stimulus if it is displayed for a span of time which is greater than the initial stimulus otherwise do not respond', 
								'if the final stimulus appears for a length of time which is greater than the first stimulus then choose the final stimulus otherwise do not respond')								

test_instruct_dict['MultiDur1'] = ('respond to the initial direction if the length of the stimulus integrated over both modalities is greater than the second direction otherwise do not respond',
									'choose the first stimulus if it spans a length of time which is greater than the second stimulus when summed over both modalities otherwise do not respond', 
									'if the first direction is displayed to a longer length of time summed over both modalities than the second direction then select the first direction otherwise do not respond',
									'select the initial orientation if it has a longer time span when integrated over both modalities than the final direction otherwise do not respond',
									'opt for the initial stimulus it it is displayed for a span of time which is longer than the span of the second stimulus summed over both modalities otherwise do not respond')

test_instruct_dict['MultiDur2'] = ('respond to the final orientation if it spans a greater period of time when summed over both modalities than the initial orientation otherwise do not respond',
									'select the last direction if it is presented for a greater length of time than the initial direction when integrated over both modalities otherwise do not respond', 
									'opt for the final stimulus if it appears for a greater span of time than the initial stimulus when summed over both modalities otherwise do not respond',
									'if the second direction is displayed for a length of time which is greater than the first direction when summed over both modalities then respond to the second direction otherwise do not respond',
									'pick the final orientation if it appears for a span of time which is larger than the first orientation when integrated over both modalities otherwise do not respond')

test_instruct_dict['AntiDur1'] = ('select the earlier stimulus if it spans a length of time which is shorter than latter stimulus otherwise do not respond',
									'opt for the earlier stimulus if it appears for a length of time which is less than the latter stimulus otherwise do not respond',
									'if the earlier direction is presented for an amount of time which is less than the latter direction than choose the earlier direction otherwise do not respond', 
									'choose the first orientation if it appears for a shorter span of time than the second stimulus otherwise do not respond',
									'pick the earlier orientation if it is displayed for a length of time that is less than the final direction otherwise do not respond')

test_instruct_dict['AntiDur2'] = ('opt for the latter stimulus if is presented for a span of time which is less than the earlier stimulus otherwise do not respond',
									'pick the final direction if it appears for a shorter length of time than the earlier direction otherwise do not respond', 
									'if the latter orientation has a shorter time span than the first orientation select the latter orientation otherwise do not respond',
									'select the latter stimulus if it has a shorter period of presentation than the first stimulus otherwise do not respond',
									'choose the latter stimulus if it is presented for a shorter length of time than the first stimulus otherwise do not respond')

test_instruct_dict['AntiMultiDur1'] = ('pick the earlier direction if its lasts for a span of time which is less than the latter direction when summed over both modalities otherwise do not respond',
											'choose the first orientation if it is presented for a smaller amount of time than the second direction when integrated over both modalities otherwise do not respond',
											'if the earlier stimulus appears for a shorter span of time when summed over both modalities than the second stimulus respond to the earlier stimulus otherwise do not respond',
											'select the initial direction if it is displayed for a length of time which is shorter than the second direction when summed over both modalities otherwise do not respond',
											'select the earlier stimulus if has a duration which is less than that of the second stimulus integrated over both modalities otherwise do not respond')

test_instruct_dict['AntiMultiDur2'] = ('pick the latter orientation if it appears for an amount of time which is less than the first orientation when summed across modalities otherwise do not respond',
										'opt for the latter direction if it is presented for a span of time which is less than the earlier orientation when integrated over both modalities otherwise do not respond',
										'respond to the latter stimulus if it appears for a length of time that is shorter than the earlier stimulus integrated across modalities otherwise do not respond',
										'if the final direction is presented for a time span that is less than the initial direction when summed over both modalities then choose the final direction otherwise do not respond',
										'if the latter stimulus appears for the shorter time span than the earlier stimulus when combined over both modalities then opt for the latter stimulus otherwise do not respond')
										
test_instruct_dict['Dur1Mod1'] = ('concentrate only on the first modality and select the earlier direction if it lasts for a longer time span than the latter direction otherwise do not respond',
									'focus exclusively on the first modality and choose the earlier stimulus if it is presented for a greater length of time than the second stimulus otherwise do not respond',
									'consider only stimuli from the first modality and pick the first stimulus if it appears for a greater period of time than the second stimulus otherwise do not respond',
									'concentrate exclusively on stimuli in the first modality and choose the first direction if it appears for a greater span of time than the second direction otherwise do not respond',
									'only consider the first modality and opt for the earlier stimulus if it appears for a length of time which is greater than the latter stimulus otherwise do not respond')


test_instruct_dict['Dur1Mod2'] = ('concentrate only on the second modality and respond to the first direction if spans a greater length of time than the second stimulus otherwise do not respond',
									'focus exclusively on the stimuli in the second modality and respond to the earlier direction if it is displayed for a greater length of time than the latter stimulus otherwise do not respond',
									'only consider stimuli in the second modality and select the earlier orientation if it appears for a greater span of time than the first orientation otherwise do not respond',
									'concentrate on the second modality and choose the initial direction if it lasts for a greater period of time than the second direction otherwise do not respond',
									'attend exclusively to the second modality and respond to the first stimulus if it is presented for a greater span of time than the final stimulus otherwise do not respond')

test_instruct_dict['Dur2Mod1'] = ('concentrate only on the first modality and select the latter orientation if it spans a longer length of time than the earlier orientation otherwise do not respond',
									'attend exclusively to the first modality and choose the latter stimulus if it last for longer than the initial stimulus otherwise do not respond',
									'only consider stimuli in the first modality and opt for the second stimulus if it spans a period of time which is greater than the first stimulus otherwise do not respond',
									'focus only on the first modality and select the latter direction if it lasts for a greater length of time than the earlier direction otherwise do not repsond',
									'concentrate on only the stimuli in the first modality and pick the final stimulus if it spans a length of time which is greater than the first stimulus otherwise do not respond')

test_instruct_dict['Dur2Mod2'] = ('focus exclusively on stimuli in the second modality and choose the final direction if it spans a period of time which is longer than the earlier direction otherwise do not respond',
									'concentrate only on the second modality and opt for the last direction if it appears for a length of time that is greater than the earlier direction otherwise do not respond',
									'only consider stimuli in the second modality and choose the final orientation if it is presented for a span of time which lasts longer than the initial orientation otherwise do not respond',
									'concentrate only on stimuli in the second modality and select the latter stimulus if it lasts for a greater length of time than the earlier stimulus otherwise do not respond',
									'attend exclusively to the second modality and opt for the final direction if it is presented for a length of time which is longer than that of the initial direction otherwise do not respond')

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

test_instruct_dict['AntiCOMP1'] = ('select the earlier orientation if it is presented with less strength than the latter orientation otherwise do not respond',
									'if the earlier direction appears less strong than the latter direction then select the earlier direction otherwise do not respond',
									'opt for the earlier stimulus if it appears weaker than the final stimulus otherwise do not respond',
									'choose the initial orientation if it is presented with lesser intensity compared to the latter stimulus otherwise do not respond',
									'opt for the earlier direction it appear with lesser intensity than the final direction otherwise do not respond'
									)

test_instruct_dict['AntiCOMP2'] = ('opt for the latter orientation if it presented with lesser strength than the earlier orientation otherwise do not respond',
									'if the second stimulus appears with less strength compared to the earlier stimulus then opt for the second stimulus otherwise do not respond',
									'choose the latter direction if it has lesser strength compared to the initial direction otherwise do not respond',
									'if the latter stimulus appears with lesser intensity compared to the earlier stimulus then pick the latter stimulus otherwise do not respond',
									'select the final direction if it is presented with less intensity compared to the initial direction otherwise do not respond')

test_instruct_dict['AntiMultiCOMP1'] = ('opt for the earlier direction if it appears with lesser intensity compared to the latter direction when summed over both modalities otherwise do not respond',
										'select the initial orientation if it is shown with lesser strength than the last orientation integrated across modalities otherwise do not respond',
										'if the first stimulus has lesser strength when compared to the latter stimulus summed over both modalities then choose the latter stimulus otherwise do not respond',
										'pick the earlier stimulus if it is presented with lower strength compared to the last stimulus when integrated over both modalities otherwise do not respond',
										'if the first direction is shown with lesser strength compared to the strength of the second direction summed over both modalities then select the first direction otherwise do not repsond')

test_instruct_dict['AntiMultiCOMP2'] = ('if the latter orientation is shown with lesser strength than the earlier orientation when added across modalities then opt for the latter orientation otherwise do not respond',
										'choose the last displayed direction if it is shown with lesser intensity compared to the first direction when integrated over modalities otherwise do not respond',
										'opt for the latter direction if it appears with less strength compared to the initial direction when stimuli strength are integrated over both modalities otherwise do not respond',
										'select the last stimulus if it is displayed with a low strength compared to the first stimulus when intensity is summed over both modalities otherwise do not respond',
										'pick the latter direction if it has less intensity compared to earlier direction when strength is integrated over both modalities otherwise do not respond')

test_instruct_dict['COMP1Mod1'] = ('concentrate only on stimuli from the first modality and opt for the first direction if it is presented with greater intensity compared to the latter direction otherwise do not respond',
									'focus exclusively on the first modality and select the earlier stimulus if is appears with a greater contrast when compared to the final stimulus otherwise do not respond',
									'only consider the first modality and if the earlier direction is presented with higher contrast than the latter direction opt for the earlier direction otherwise do not respond',
									'exclusively attend to stimuli from the first modality and respond to the earlier orientation if it is displayed with higher intensity compared to the last orientation otherwise do not respond',
									'concentrate only on the first modality and pick the earlier direction if it is displayed with higher contrast compared to the latter direction otherwise do not respond')

test_instruct_dict['COMP1Mod2'] = ('only consider stimuli from the second modality and select the earlier direction if it appears with a higher contrast than the latter direction otherwise do not respond',
									'concentrate exclusively on the second modality and choose the earlier orientation if it is displayed with a greater intensity than the last orientation otherwise do not respond',
									'focus entirely on the second modality and if the initial stimulus is displayed with higher contrast than the last stimulus respond to the first stimulus otherwise do not respond',
									'consider only stimuli from the second modality and if the initial direction has a greater contrast when compared to the last direction then select the initial direction otherwise do not respond',
									'attend exclusively to the second modality and if the first orientation is shown with a higher strength compared to the final direction then pick the first orientation otherwise do not respond')

test_instruct_dict['COMP2Mod1'] = ('only consider stimuli from the first modality and opt for the last direction if it appears with a higher contrast than the initial direction otherwise do not respond',
									'focus solely on the stimuli from the first modality and choose the latter stimulus if it is presented with less strength compared to that of the earlier stimulus otherwise do not respond',
									'concentrate only on the first modality and if the latter orientation appears stronger compared to the initial orientation then opt for the latter orientation otherwise do not respond', 
									'attend exclusively to the first modality and select the final direction if it is presented with greater contrast compared to that of the earlier direction otherwise do not respond',
									'consider only the first modality and respond to the last stimulus if is has a contrast which is higher compared to the contrast of the initial stimulus otherwise do not respond')

test_instruct_dict['COMP2Mod2'] = ('consider only stimuli from the second modality and choose the last orientation if it is displayed with a contrast that is higher than the contrast of the earlier orientation otherwise do not respond', 
									'attend solely to the second modality and select the latter direction if it appears with a greater strength compared to that of the earlier direction otherwise do not respond', 
									'concentrate only on stimuli from the second modality and if the last stimulus has a larger contrast compared to that of the earlier stimulus then select the last stimulus otherwise do not respond',
									'focus exclusively on the second modality and if the last direction is displayed with a higher contrast compared to the initial direction respond to the last direction otherwise do not respond',
									'only consider the second modality and pick the latter direction if it is displayed with greater intensity compared to the earlier direction otherwise do not respond')

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



def save_instruct_dicts(models_path):
	path = models_path+'/instructs/'
	if os.path.exists(path):
		pass
	else: 
		os.makedirs(path)

	#pickle.dump(train_instruct_dict, open(path+'train_instruct_dict', 'wb'))
	pickle.dump(test_instruct_dict, open(path+'test_instruct_dict', 'wb'))


