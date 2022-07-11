import pymysql

class Database:
    def connect():
        return pymysql.connect(host='localhost',
                             user='rischan',
                             password='1234',
                             db='rischandb')