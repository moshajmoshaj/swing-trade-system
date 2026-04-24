def calc_position_size(
    capital: float, atr: float, price: float
) -> tuple[int, float]:
    """
    Returns:
        shares: 購入株数
        stop_price: 損切り価格
    """
    stop_width = atr * 2
    stop_price = price - stop_width

    by_risk    = int((capital * 0.02) / stop_width)
    by_max_pos = int(200_000 / price)
    by_unit    = 100

    shares = min(by_risk, by_max_pos, by_unit) if by_unit > 0 else 0
    shares = max(shares, 0)

    return shares, stop_price
