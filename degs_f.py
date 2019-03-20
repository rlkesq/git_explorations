#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'git_explore'))
	print(os.getcwd())
except:
	pass

#%%
# Change directory to VSCode workspace root so that relative path loads work correctly. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..'))
	print(os.getcwd())
except:
	pass



#%%

def future_value_investment():
    principal = eval(input('Enter the initial principal amount: '))
    ann_yield = eval(input('Enter the annual yield on the investment'))
    years = eval(input('Enter the period of the investment (years)'))
    closing_balance = principal
    for n in range(years):
        closing_balance = closing_balance * (1 + ann_yield)
        print ('year ', n, 'closing balance: ', closing_balance)

future_value_investment()
#%%
print('start')
for i in range(0):
    print(i)
print('end')