import json
import re
import math  


def range_reward(x: float, low: float, high: float) -> float:
    """
    改进版：区间内奖励更平缓，避免"随便落在区间里就行"
    - 在区间内：中间最好 (=1.0)，边界 (=0.5)
    - 超出区间：按距离扣分
    
    返回值范围：约 [-1, 1]
    """
    if x < low:
        return -min((low - x) / (high - low), 1.0)
    elif x > high:
        return -min((x - high) / (high - low), 1.0)
    else:
        mid = 0.5 * (low + high)
        half = 0.5 * (high - low)
        return 0.5 + 0.5 * (1.0 - abs(x - mid) / half)


def parse_action(response: str):
    """解析模型输出的 JSON"""
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except:
        return None


def _to_float_or_none(x):
    """将值转换为 float，如果是 None/NaN/Inf 则返回 None"""
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


_DEBUG_COUNT = 0
_DEFAULT_VALUE_COUNT = 0  # 追踪默认值触发次数


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    """
    verl reward 函数 - 微观约束 + 宏观局部约束
    
    评分维度（权重总和 = 1.0）：
    1. JSON 格式正确性
    2. 储蓄率约束 [0.014, 0.318]（权重 0.10）
    3. Work 行为偏好（权重 0.25）
    4. 宏观约束（权重 0.65）
    """
    global _DEBUG_COUNT, _DEFAULT_VALUE_COUNT
    _DEBUG_COUNT += 1
    
    # 调试日志（只记录前 5 个）
    if _DEBUG_COUNT <= 5:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
                f.write(f"solution_str: {repr(solution_str)[:500]}\n")
                f.write(f"extra_info: {repr(extra_info)[:300]}\n")
        except:
            pass
    
    if data_source != "econ_agent":
        return 0.0
    
    reward = 0.0
    
    # ========== 1. JSON 格式检查 ==========
    action = parse_action(solution_str)
    if action is None:
        return -1.0
    
    work = action.get("work")
    consumption = action.get("consumption")
    
    if work is None or consumption is None:
        return -0.8
    
    try:
        work = float(work)
        consumption = float(consumption)
    except:
        return -0.8
    
    # ========== 2. 范围检查（软约束）==========
    if not (0 <= work <= 1):
        reward -= 0.3
    if not (0 <= consumption <= 1):
        reward -= 0.3
    
    work = max(0.0, min(1.0, work))
    consumption = max(0.0, min(1.0, consumption))
    
    # ========== 3. 解析 extra_info ==========
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif extra_info is None:
        extra_info = {}
    
    # ------ 3.1 微观变量（与分析脚本一致）------
    # DPI = income + lump_sum - tax_paid
    income   = _to_float_or_none(extra_info.get('income', 0))   or 0.0
    lump_sum = _to_float_or_none(extra_info.get('lump_sum', 0)) or 0.0
    tax_paid = _to_float_or_none(extra_info.get('tax_paid', 0)) or 0.0
    wealth   = _to_float_or_none(extra_info.get('wealth', 0))   or 0.0

    
    dpi = _to_float_or_none(extra_info.get('dpi', None))
    if dpi is None:
        dpi = income + lump_sum - tax_paid
    dpi = max(dpi, 1.0)  # 避免除零

    buffer_ratio = _to_float_or_none(extra_info.get('buffer_ratio', None))
    if buffer_ratio is None:
        cash_on_hand = wealth + dpi
        buffer_ratio = cash_on_hand / dpi

    
    # ------ 3.2 宏观变量 ------
    # unemployment_rate: 小数，如 0.08 表示 8%
    # gdp_growth: 百分比，如 1.5 表示 1.5%
    # price_inflation: 百分比，如 2.0 表示 2%
    # 放文件顶部一次即可

    unemp = _to_float_or_none(extra_info.get('unemployment_rate', None))
    gdp_g = _to_float_or_none(extra_info.get('gdp_growth', None))
    infl  = _to_float_or_none(extra_info.get('price_inflation', None))


    
    # ========== 4. Saving Rate 计算 ==========
    # 与分析脚本一致: saving_rate = (DPI - consumption_amount) / DPI
    consumption_amount = consumption * dpi
    saving = dpi - consumption_amount
    saving_rate = saving / dpi  # 等于 1 - consumption
    
    # ========== 5. 评分计算（权重总和 = 1.0）==========
    
    # ------ 5.1 储蓄率约束（权重 0.10）------
    # 目标区间：1.4% ≤ saving_rate ≤ 31.8%
    sr_reward = range_reward(saving_rate, 0.014, 0.60)  # 允许储蓄率 1.4%-60%
    reward += 0.10 * sr_reward

    
    # ------ 5.2 Work 行为偏好（权重 0.25）------
    work_reward = 0.0
    
    # 钱少（buffer_ratio < 2）应该多干活
    if buffer_ratio < 2:
        if work >= 0.8:
            work_reward = 0.6
        elif work >= 0.6:
            work_reward = 0.5
        elif work < 0.3:
            work_reward = -0.5  # 钱少还不工作，惩罚
        else:
            work_reward = 0.0
    
    # 钱多（buffer_ratio > 4）可以少工作
    elif buffer_ratio > 4:
        if work <= 0.3:
            work_reward = 0.5  # 钱多休息是合理的
        elif work >= 0.8:
            work_reward = -0.2  # 钱多还努力工作也OK
        else:
            work_reward = 0.2
    
    # 中等情况（2 <= buffer_ratio <= 4）
    else:
        if 0.5 <= work <= 0.9:
            work_reward = 0.5
        elif work >= 0.3:
            work_reward = 0.2
        else:
            work_reward = -0.2
    if buffer_ratio > 2 and work > 0.9:
        work_reward -= 0.1 * (work - 0.9) / 0.1   # work=1.0 时额外 -0.1

    reward += 0.25 * work_reward
    
    # ------ 5.3 宏观约束（权重 0.65）：regime-conditioned soft targets ------
    # 核心：不模仿 gt 行为；只奖励"在该宏观环境下的合理反应区间"
    # reward 完全由 action 决定（可归因），且用区间奖励避免推向 0/1 极端

    macro_reward = 0.0

    regime = extra_info.get("regime", None)
    if regime is None:
        regime = "normal"
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = _to_float_or_none(extra_info.get("regime_strength", None))
    if regime_strength is None:
        regime_strength = 0.15
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime_strength at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = max(0.0, min(1.0, regime_strength))

    # ========== (A) regime soft targets on actions ==========
    # 用 range_reward 做"软区间"，区间内中间最好，边界也还行（避免极端化）
    # 这些阈值是"方向优先"的初始值：你可以后续基于仿真分布再调

    if regime == "recession":
        # 衰退：偏高劳动、偏低消费（与储蓄率约束兼容）
        work_r = range_reward(work, 0.65, 0.95)
        cons_r = range_reward(consumption, 0.40, 0.65)
        action_struct = 0.6 * work_r + 0.4 * cons_r

    elif regime == "boom":
        # 繁荣：允许更高消费、较低劳动
        work_r = range_reward(work, 0.10, 0.45)
        cons_r = range_reward(consumption, 0.75, 0.95)
        action_struct = 0.5 * work_r + 0.5 * cons_r

    else:
        # 正常：鼓励中庸（与储蓄率约束兼容）
        work_r = range_reward(work, 0.35, 0.75)
        cons_r = range_reward(consumption, 0.55, 0.80)
        action_struct = 0.5 * work_r + 0.5 * cons_r

    # ========== (B) anti-extreme brake (very light) ==========
    # 防止学会永远输出 0 或 1
    extreme_pen = 0.0
    if work < 0.05 or work > 0.95:
        extreme_pen -= 0.15
    if consumption < 0.05 or consumption > 0.95:
        extreme_pen -= 0.15

    # ========== (C) optional macro guardrail (very small) ==========
    # 只防爆，不贴区间，不压平
    guard_parts = []
    guard_w = []

    if unemp is not None:
        guard_parts.append(range_reward(unemp, 0.02, 0.20))
        guard_w.append(0.40)

    if gdp_g is not None:
        guard_parts.append(range_reward(gdp_g, -5.0, 10.0))
        guard_w.append(0.35)

    if infl is not None:
        guard_parts.append(range_reward(infl, -2.0, 8.0))
        guard_w.append(0.25)

    if guard_w:
        wsum = sum(guard_w)
        guard = sum(p * w for p, w in zip(guard_parts, guard_w)) / wsum
    else:
        guard = 0.0

    guard = max(-1.0, min(1.0, guard))

    # ========== (D) combine ==========
    # 结构为主，护栏为辅；regime_strength 控制"明显衰退/繁荣"更有训练信号
    macro_reward = (0.9 * action_struct + 0.1 * guard + extreme_pen) * regime_strength

    reward += 0.65 * macro_reward

    # ========== 6. 调试日志 ==========
    if _DEBUG_COUNT <= 20:
        try:
            unemp_str = f"{unemp:.3f}" if unemp is not None else "None"
            gdp_str = f"{gdp_g:.2f}" if gdp_g is not None else "None"
            infl_str = f"{infl:.2f}" if infl is not None else "None"
            # 显示宏观指标使用了哪些项（诊断友好）
            macro_used = ''.join([
                'pi' if infl is not None else '',
                'u'  if unemp is not None else '',
                'g'  if gdp_g is not None else ''
            ]) or 'none'
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[{_DEBUG_COUNT:03d}] "
                    f"sr={saving_rate:.3f} sr_r={sr_reward:.3f} | "
                    f"buf={buffer_ratio:.2f} work={work:.2f} work_r={work_reward:.2f} | "
                    f"unemp={unemp_str} gdp={gdp_str} infl={infl_str} macro_used={macro_used} macro_r={macro_reward:.2f} | "
                    f"w_sr={0.10*sr_reward:.3f} w_work={0.25*work_reward:.3f} w_macro={0.65*macro_reward:.3f} | "
                    f"total={reward:.3f}\n"
                )
        except:
            pass
    # 每 1000 个样本报告一次默认值触发情况
    if _DEBUG_COUNT % 1000 == 0 and _DEFAULT_VALUE_COUNT > 0:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"[STATS] Processed {_DEBUG_COUNT} samples, default value triggered {_DEFAULT_VALUE_COUNT} times ({_DEFAULT_VALUE_COUNT/_DEBUG_COUNT*100:.2f}%)\n")
        except:
            pass    
    # ========== 7. 最终 clip ==========
    reward = max(-1.0, min(1.0, reward))
    
    return reward
