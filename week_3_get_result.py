        testTargetGenderArray.append(testTargetGender)

        row_count += 1


    row_count1 = 0
# open test data csv file
with open(csv_test_file_Age) as csvtestfileAge:
    readTestCSVage = csv.reader(csvtestfileAge, delimiter = ',')

     # skip the first row
    next(csvtestfileAge)

    # row count is to count the total of test target
    row_count = 0

    for row1 in readTestCSVage:
        testTargetID = row1[1]
        testTargetAge = row1[2]
        if(testTargetAge == '1'):
            #print('1')
            testTargetAge = 'xx-24'
        elif(testTargetAge == '2'):
            testTargetAge = '25-34'
            #print('2')
        elif(testTargetAge == '3'):
            testTargetAge = '35-49'
            #print('3')
        elif(testTargetAge == '4'):
            testTargetAge = '50-xx'
            #print('4')

        #print(testTargetAge)
        testTargetIDArray.append(testTargetID)
        testTargetAgeArray.append(testTargetAge)
        row_count1 += 1

     # output xml file
for row in range(row_count1):
        root = et.Element('user', id=testTargetIDArray[row], age_group=testTargetAgeArray[row],
                          gender=testTargetGenderArray[row], extrovert='3.49',
                          neurotic='2.73', agreeable='3.58',
                          conscientious='3.45', open='3.91')

        tree = et.ElementTree(root)
        # folder_name = testTargetIDArray[row] + ".xml"
        # tree.write(folder_name)
        folder_name = output_path + testTargetIDArray[row] + '.xml'
        tree.write(folder_name)

        # move the file into a folder called result
        # os.rename(folder_name, base_path +"/result/"+ folder_name)

os.remove(str(Path.home()) + "/profile_test_with_gender_prediction.csv")
os.remove(str(Path.home()) + "/profile_test_with_Age_prediction.csv")
#print('Done')
csvtestfile.close()


