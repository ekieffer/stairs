class CashShortage(Exception):
   """
   Raise an exception if cash shortage (i.e. contributions cannot be honored)
   """

   def __init__(self, cash):
     self.cash = cash
     super().__init__("")

   def __str__(self):
     return f'{self.cash} -> shortage'


class CashFlows_Freq(Exception):
    """
    Raise an exception if cash shortage (i.e. contributions cannot be honored)
    """

    def __init__(self, message):
        self.message = message
        super().__init__("")

    def __str__(self):
        return f'{self.message}'
