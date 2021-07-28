

def Capalize(arr):


    for i in arr:
        if type(i) =='str':
            print( i.capitalize() )


def vowels_count(string):

    
    obj = {
        'a':0,
        'e':0,
        'i':0,
        'o':0,
        'u':0
    }

     
    for i in string:
        try :
            if ( obj[i] >=0 ):
                
                obj[i] = obj[i] + 1;
        except KeyError:
            continue


    print('Count of a = ',obj['a'])
    print('Count of e = ',obj['e'])
    print('Count of i = ',obj['i'])
    print('Count of o = ',obj['o'])
    print('Count of u = ',obj['u'])

    return vowels_count;



def second_largest(arr):

    # sorted_arr = sorted(arr,reverse=True)

    if ( ( arr[0] < arr[1] or arr[1] > arr[2] )  and ( arr[0] > arr[1] or arr[1] < arr[2] ) ):
        print('second largest => ',arr[1],' offset => ',1)
    elif ( (arr[1] < arr[0] or arr[2] > arr[2] ) and ( arr[1] > arr[0]  or arr[0] < arr[2]) ):
        print('second largest => ',arr[0],' offset => ',0)
    elif ( ( arr[1] < arr[2] or arr[0] < arr[2] ) and ( arr[2] < arr[1] or arr[2] < arr[0]) ):
        print('second largest => ',arr[2],' offset => ',2)   


arr = [5,5,2]
# second_largest(arr);

'''
102
3 per choco
4 wrapper 1 free 

34 * 3 == 102

free 34 % 4 = 8

3x + ( x % 4 ) * 1  = 102
'''

vowels_count('apple')