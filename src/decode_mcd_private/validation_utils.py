from decode_mcd.mcd_exceptions import UserInputException


def validate(mandatory_condition: bool, exception_message: str):
    if not mandatory_condition:
        raise UserInputException(exception_message)
