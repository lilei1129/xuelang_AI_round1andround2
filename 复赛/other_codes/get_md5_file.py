# -*- coding: utf-8 -*-
###by feifei####
import hashlib
import sys
class GETMD5(object):
    def isFile(self,path):
        return True
    def getFileMD5(self,filepath):
        if self.isFile(filepath):
            f = open(filepath,'rb')
            md5obj = hashlib.md5()
            md5obj.update(f.read())
            hash = md5obj.hexdigest()
            f.close()
            return str(hash).upper()
        return None
    #获取文件的MD5值，适用于较大的文件
    def getBigFileMD5(self,filepath):
        if self.isFile(filepath):
            md5obj = hashlib.md5()
            maxbuf = 8192
            f = open(filepath,'rb')
            while True:
                buf = f.read(maxbuf)
                if not buf:
                    break
                md5obj.update(buf)
            f.close()
            hash = md5obj.hexdigest()
            return str(hash).upper()
        return None
if __name__ == '__main__':
    a=GETMD5()
    s2=a.getBigFileMD5('./model/cut320.h5')
#    if len(sys.argv)!= 2:
#        sys.exit('argv error!')
# 
#    m = hashlib.md5()
#    n = 1024*4
#    inp = open(sys.argv[1],'rb')
#    while True:
#        buf = inp.read(n)
#        if buf:
#            m.update(buf)
#        else:
#            break
#    print(m.hexdigest())
#获取文件的MD5值，适用于小文件
