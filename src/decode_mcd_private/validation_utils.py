from decode_mcd.mcd_exceptions import UserInputException


def validate(mandatory_condition: bool, exception_message: str, warning: bool = False):
    if not mandatory_condition:
        if warning:
            print(f"Warning: {exception_message}")
        else:
            raise UserInputException(exception_message)
