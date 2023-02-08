class Image(object):
    def __init__(self, label, features):
        self.class_label = label
        self.class_features = features

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
