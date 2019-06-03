# A Python-3 adapted version of the code from
# http://code.activestate.com/recipes/81611-roman-numerals/#c3

def roman_to_int(input):
   """
   Convert a roman numeral to an integer.

   >>> r = range(1, 4000)
   >>> nums = [int_to_roman(i) for i in r]
   >>> ints = [roman_to_int(n) for n in nums]
   >>> print r == ints
   1

   >>> roman_to_int('VVVIV')
   Traceback (most recent call last):
    ...
   ValueError: input is not a valid roman numeral: VVVIV
   >>> roman_to_int(1)
   Traceback (most recent call last):
    ...
   TypeError: expected string, got
   >>> roman_to_int('a')
   Traceback (most recent call last):
    ...
   ValueError: input is not a valid roman numeral: A
   >>> roman_to_int('IL')
   Traceback (most recent call last):
    ...
   ValueError: input is not a valid roman numeral: IL
   """
   try:
      input = input.upper()
   except AttributeError:
      raise ValueError('expected string, got %s' % type(input))
   # map of (numeral, value, maxcount) tuples
   roman_numeral_map = (('M',  1000, 3), ('CM', 900, 1),
                        ('D',  500, 1), ('CD', 400, 1),
                        ('C',  100, 3), ('XC', 90, 1),
                        ('L',  50, 1), ('XL', 40, 1),
                        ('X',  10, 3), ('IX', 9, 1),
                        ('V',  5, 1),  ('IV', 4, 1), ('I',  1, 3))
   result, index = 0, 0
   for numeral, value, maxcount in roman_numeral_map:
      count = 0
      while input[index: index +len(numeral)] == numeral:
         count += 1 # how many of this numeral we have
         if count > maxcount:
            raise ValueError('input is not a valid roman numeral: %s' % input)
         result += value
         index += len(numeral)
   if index < len(input): # There are characters unaccounted for.
      raise ValueError('input is not a valid roman numeral: %s'%input)
   else:
      return result


def roman_to_int_wrapper(s):
  '''
  Convert a roman numeral to an integer.

  :param s: input string
  :return: A simple wrapper that returns None when no valid number is given as input.
  '''
  try:
    return roman_to_int(s)
  except:
    return None
