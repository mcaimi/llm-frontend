#!/usr/bin/env python

class Parameters(object):
    def __init__(self, data: dict):
        if type(data) != dict:
            raise TypeError(f"Parameters: expected 'dict', got {type(data)}.")
        else:
            self.data = data

        for k in self.data.keys():
            if type(self.data.get(k)) != dict:
                self.__setattr__(k, self.data.get(k))
            else:
                self.__setattr__(k, Parameters(self.data.get(k)))
