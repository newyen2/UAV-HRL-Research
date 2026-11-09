"""Presentation-only numbers; editable settings keep their original precision."""
import json
from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext


# 依數值大小截斷顯示位數，格式化介面中的數字。
def format_number(value):
    """Truncate toward zero with magnitude-dependent decimal places."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return "—"
    if not number.is_finite():
        return "—"
    magnitude = abs(number)
    with localcontext() as context:
        # 保留足夠精度，避免截斷高精度數字時觸發 Decimal 錯誤。
        context.prec = max(32, len(number.as_tuple().digits) + 8)
        # 很小但非零的指標也用科學記號，避免在圖表與表格中被誤讀為 0。
        if magnitude >= Decimal("100000") or 0 < magnitude < Decimal("0.0001"):
            exponent = magnitude.adjusted()
            mantissa = number.scaleb(-exponent).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
            return f"{mantissa:.4f}e{exponent:+03d}"
        if magnitude < 10:
            places = 4
        elif magnitude < 100:
            places = 3
        elif magnitude < 1000:
            places = 2
        elif magnitude < 10000:
            places = 1
        else:
            places = 0
        truncated = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)
        if not truncated:
            truncated = abs(truncated)
        return f"{truncated:.{places}f}"


# 展開科學記數法並保留設定值的完整精度。
def format_edit_value(value):
    """Expand exponent notation without truncating hyperparameter values."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_edit_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(str(key), ensure_ascii=False) + ": "
                               + format_edit_value(item) for key, item in value.items()) + "}"
    try:
        number = Decimal(str(value))
        if number.is_finite():
            return format(number, "f")
    except (InvalidOperation, ValueError, TypeError):
        pass
    return str(value)
