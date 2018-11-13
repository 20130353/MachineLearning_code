# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 18-11-8
# file: test
# description:

def solution(str):
  stack = []
  res = [0]
  for each in str:
      if each == '(':
        stack.append(each)
      elif each == ')':
          if len(stack) != 0:
            top = stack.pop()
            if top != '(':
              res.append(0)
              stack.append(top)  #  放回去，以免漏匹配
            else:
              res.append(res[-1]+1)  # 匹配正确+1

      else:
        print('unknown char!')
      return max(res)


if __name__ == '__main__':
  str = '(()'
  str1 = '()())'
  str2 ='fdasfas()'
  str3 = '()('

  res = solution(str)
  print(res)
