def main():
    print ('This program computes the average of two exam scores')
    score1, score2 = eval(input('Enter two scores separated by a comma: '))
    print('The two entries are of types: ',type(score1), type(score2))
    average = (score1 + score2)/2
    print ('The average of the scores is: ', average)

main()
