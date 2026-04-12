#!/usr/bin/env python3
"""
Generate Chinese e-commerce customer-service golden datasets with DeepSeek.

This script deliberately keeps control logic local and lets the model focus on
writing the final customer-service reply. That makes the output more stable than
asking the model to invent the entire schema object from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "doc"
ENV_PATH = ROOT / ".env"

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = (
    "你是电商客服助手，需要礼貌、专业、基于已知信息回答，不编造订单状态、物流进度、处理结果"
    "或承诺时间；信息不足时先追问，超出权限或高风险场景时转人工。"
)
FORBIDDEN_PROMISES = [
    "今天一定发",
    "明天就能到",
    "已经帮您修改成功",
    "已经为您取消成功",
    "已经退款成功",
    "仓库今天一定会处理",
    "我帮您操作好了",
    "已经加急处理完成",
]
FORBIDDEN_PHRASES = [
    "根据平台规则",
    "平台规定",
]
DECISION_TYPES = {
    "direct_answer",
    "ask_followup",
    "rule_explanation",
    "transfer_human",
}
ORDER_ID_REQUEST_HINTS = ["订单号", "手机号尾号", "订单信息"]
TRANSFER_HINTS = ["人工", "人工客服"]


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    scenario: str
    context: dict[str, Any]
    history: list[dict[str, str]]
    instruction: str
    intent: str
    decision_type: str
    should_transfer: bool
    missing_slots: list[str]
    policy_basis: list[str]
    risk_flags: list[str]


# Train/eval use different utterance pools so the eval set stays more disjoint.
CANCEL_UNSHIPPED_TRAIN = [
    "帮我取消订单",
    "这个订单我不想要了，麻烦取消一下",
    "还没发货的话帮我取消吧",
    "我下错单了，想取消订单",
    "这个订单现在能取消吗？",
    "不要发了，帮我取消这个订单",
    "刚买错了，帮我撤掉订单",
    "这单先别发，帮我取消",
]
CANCEL_UNSHIPPED_EVAL = [
    "这笔订单我现在不需要了，能直接取消吗？",
    "还没出库的话，请帮我判断一下能不能取消",
    "买重复了，这个订单还有办法取消吗？",
    "如果还没发出，我想现在就取消订单",
]
CANCEL_SHIPPED_TRAIN = [
    "我想取消这个订单，可以吗？",
    "已经发货了还能取消吗？",
    "这个包裹在路上了，我还能撤单吗？",
    "订单发出了，我现在不想要了，能取消吗？",
    "快递已经在运输中了，还能帮我取消吗？",
    "已经发货的话是不是不能取消了？",
    "包裹已经寄出了，还可以取消订单吗？",
    "这单发货了，我还能现在取消吗？",
]
CANCEL_SHIPPED_EVAL = [
    "订单已经发出去了，我现在还能不能取消？",
    "物流都动了，这种情况还能撤单吗？",
    "东西在路上了，但我不想要了，可以直接取消吗？",
    "已经发货的订单想取消，该怎么处理？",
]
CANCEL_UNKNOWN_TRAIN = [
    "我想取消订单，帮我看下能不能取消",
    "这个订单我要取消",
    "帮我取消一下这单",
    "我不想要了，取消订单吧",
    "这笔订单能不能取消，你先帮我判断下",
    "取消这单之前，麻烦先帮我看下现在还支不支持",
]
CHANGE_UNSHIPPED_TRAIN = [
    "我想修改一下收货地址，现在还没发货吧？",
    "订单还没发的话可以改地址吗？",
    "帮我看下这单没发货的话能不能换地址",
    "这单地址填错了，未发货的话可以改吗？",
    "我收货地址写错了，现在还能改吗？",
    "还没发货的话，地址可以改一下吗？",
]
CHANGE_UNSHIPPED_EVAL = [
    "这单还没发出去的话，我想改个地址",
    "未发货状态下可以调整收货地址吗？",
    "如果仓库还没发货，我现在能不能改地址？",
    "收件地址填错了，没发货的话怎么改？",
]
CHANGE_COLLECTED_TRAIN = [
    "我地址填错了，现在能改吗？",
    "订单已经发出了，地址还能修改吗？",
    "物流都揽收了，我还能改收货地址吗？",
    "快递已经在路上了，收货地址现在还能动吗？",
    "地址写错了，但包裹已经寄出，怎么办？",
    "已揽收的订单还能改地址吗？",
]
CHANGE_COLLECTED_EVAL = [
    "包裹都被物流收走了，现在还能改地址吗？",
    "这单已经交给快递了，还能不能改收件地址？",
    "物流显示已揽收，这时候地址还能调整吗？",
    "已经发货的订单想改地址，该怎么处理？",
]
CHANGE_UNKNOWN_TRAIN = [
    "我想改地址，帮我处理一下",
    "收货地址填错了，麻烦给我改下",
    "帮我换个收货地址",
    "这单地址写错了，想修改",
]
URGE_NO_ORDER_TRAIN = [
    "我的订单怎么还没发货？都等了好几天了，真的很着急，能不能快点发？",
    "怎么一直不发货，我这边很急，麻烦催一下",
    "下单这么久了还没发，能不能帮我催催？",
    "订单迟迟不发货，我着急用，麻烦看下",
    "为什么还没发货，我这边已经等很久了",
    "能不能赶紧发货，我真的挺急的",
]
URGE_NO_ORDER_EVAL = [
    "还不发货吗？我已经等很久了，能帮我催下吗？",
    "这单怎么一点动静都没有，我挺着急的，快帮我看看",
    "下单后一直没发货，能不能帮我反馈催发？",
    "我急着用，为什么到现在还没发？",
]
URGE_KNOWN_TRAIN = [
    "订单还没发货，能不能帮我催一下？",
    "这单显示未发货，麻烦帮我反馈催发",
    "还没出库的话，能帮我催一催吗？",
    "未发货的订单可以帮忙催发吗？",
    "这单一直没发，我很着急，麻烦催一下",
    "能帮我催发货吗？订单现在还是未发货",
]
URGE_TRANSFER_TRAIN = [
    "我已经催过好多次了还是没发货，你们到底怎么处理的？现在马上给我转人工。",
    "这单一直不发，我已经很不满意了，别机器人回复我，转人工。",
    "我催了几次都没结果，今天必须给我人工处理。",
    "一直敷衍我发货问题，麻烦现在转人工客服。",
]
LOGISTICS_TRANSIT_TRAIN = [
    "我的包裹怎么还没到？物流显示运输中好几天了。",
    "物流一直是运输中，为什么还没送到？",
    "快递卡在运输中了，现在什么情况？",
    "包裹怎么还在路上，一直没到。",
    "物流显示运输中，但我迟迟没收到，怎么回事？",
    "一直运输中还没到，我想知道现在什么情况",
]
LOGISTICS_TRANSIT_EVAL = [
    "物流还在运输中，为什么这么久都没送达？",
    "包裹一直在路上，我现在想知道进展",
    "运输中好多天了，到底什么时候能送到？",
    "我的件怎么还没到，物流始终显示运输中",
]
LOGISTICS_UNKNOWN_TRAIN = [
    "帮我查一下物流到哪了",
    "我想看物流进度",
    "快递现在到哪里了？",
    "帮我查查这个订单的物流",
]
LOGISTICS_STALLED_TRAIN = [
    "物流三天没更新了，我有点着急，这正常吗？",
    "包裹已经好几天没动静了，麻烦看下怎么回事",
    "快递卡了几天都不更新，我想知道现在什么情况",
    "物流停了很久，我担心是不是出问题了",
    "物流一直没刷新，我有点担心，能帮我看看吗？",
    "这单物流停了三天还没更新，是不是异常了？",
]
LOGISTICS_STALLED_TRANSFER_TRAIN = [
    "物流五天没更新了，我已经投诉过一次了，还是没人管，给我转人工。",
    "快递一直不动，我怀疑丢件了，现在要人工处理。",
    "物流这么多天没变化，我已经非常不满了，转人工。",
    "包裹停滞太久了，我需要人工跟进，不要再机器人回复。",
]
REFUND_GENERIC_TRAIN = [
    "我要退款",
    "这个订单我想申请退款",
    "帮我看下怎么退款",
    "我现在想退掉，应该怎么申请？",
    "这单我想退，麻烦告诉我退款怎么走",
    "订单不想要了，退款流程怎么操作？",
]
REFUND_GENERIC_EVAL = [
    "我想处理退款，下一步怎么走？",
    "这个订单想退款，你帮我判断下流程",
    "我现在需要退款，应该选哪种方式？",
    "这单退款怎么申请？",
]
RETURN_UNOPENED_TRAIN = [
    "商品收到了，还没拆封，现在想退货可以吗？",
    "这个东西我没拆开，想退掉，能处理吗？",
    "已签收但没拆封，可以申请退货退款吗？",
    "商品未拆封，我现在不想要了，能退吗？",
]
RETURN_OPENED_QUALITY_YES_TRAIN = [
    "我买的商品有质量问题，已经拆封了，能退货吗？",
    "东西拆开后发现有瑕疵，这种情况还能售后吗？",
    "商品用了下发现有问题，已经开封了，怎么办？",
    "拆封后发现质量有问题，还能申请退货退款吗？",
]
RETURN_OPENED_QUALITY_NO_TRAIN = [
    "这个商品我已经拆封了，但没什么问题，想退货。",
    "东西开封了，也没有质量问题，现在还能退吗？",
    "商品拆开看过了，没问题，但我不想要了，可以退吗？",
    "已经拆封且没有质量问题，这种能退货吗？",
]
RETURN_MISSING_INFO_TRAIN = [
    "这个商品我想退掉",
    "帮我判断下这个订单能不能退货",
    "我现在想申请售后，能处理吗？",
    "这个东西不想要了，能退吗？",
]
TRANSFER_EXPLICIT_TRAIN = [
    "别机器人回复了，我要人工客服",
    "这个问题我想直接找人工处理",
    "麻烦帮我转人工客服",
    "我现在只想和人工沟通",
]


def add_cases(
    bank: list[CaseSpec],
    slug_prefix: str,
    scenario: str,
    utterances: list[str],
    instruction: str,
    intent: str,
    decision_type: str,
    should_transfer: bool,
    missing_slots: list[str],
    policy_basis: list[str],
    risk_flags: list[str],
    base_context: dict[str, Any],
    *,
    multi_turn: bool = False,
) -> None:
    for index, utterance in enumerate(utterances, start=1):
        history = [{"role": "user", "content": utterance}]
        if multi_turn:
            history = [
                {"role": "user", "content": "前面这个问题一直没处理好。"},
                {"role": "assistant", "content": "抱歉给您带来不便，请您再描述一下当前情况。"},
                {"role": "user", "content": utterance},
            ]
        bank.append(
            CaseSpec(
                slug=f"{slug_prefix}_{index:03d}",
                scenario=scenario,
                context=dict(base_context),
                history=history,
                instruction=instruction,
                intent=intent,
                decision_type=decision_type,
                should_transfer=should_transfer,
                missing_slots=list(missing_slots),
                policy_basis=list(policy_basis),
                risk_flags=list(risk_flags),
            )
        )


def build_case_bank(split: str) -> list[CaseSpec]:
    bank: list[CaseSpec] = []
    if split == "train":
        add_cases(
            bank,
            "train_cancel_unshipped",
            "取消订单",
            CANCEL_UNSHIPPED_TRAIN,
            "订单未发货，可直接说明可以取消并给操作路径，不要假装已经取消成功。",
            "询问未发货订单是否可取消",
            "direct_answer",
            False,
            [],
            ["未发货可取消"],
            [],
            {
                "user_request": "取消订单",
                "shipping_status": "未发货",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_cancel_shipped",
            "取消订单",
            CANCEL_SHIPPED_TRAIN,
            "订单已发货，不可取消，应引导退货，不能承诺拦截成功。",
            "询问已发货订单是否可取消",
            "direct_answer",
            False,
            [],
            ["已发货不可取消应引导退货"],
            [],
            {
                "user_request": "取消订单",
                "shipping_status": "已发货",
                "logistics_status": "运输中",
                "is_signed": "否",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_cancel_unknown",
            "取消订单",
            CANCEL_UNKNOWN_TRAIN,
            "发货状态未知，先追问最少必要信息，不要直接判断能否取消。",
            "取消订单但发货状态未知",
            "ask_followup",
            False,
            ["shipping_status"],
            ["取消订单状态未知先追问"],
            ["信息不足"],
            {
                "user_request": "取消订单",
                "shipping_status": "未知",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_change_unshipped",
            "修改地址",
            CHANGE_UNSHIPPED_TRAIN,
            "订单未发货，可修改地址，直接给路径或下一步，不要说已经修改成功。",
            "询问未发货订单是否可修改地址",
            "direct_answer",
            False,
            [],
            ["未发货可修改地址"],
            [],
            {
                "user_request": "修改地址",
                "shipping_status": "未发货",
                "new_address_provided": "是",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_change_collected",
            "修改地址",
            CHANGE_COLLECTED_TRAIN,
            "订单已发货且已揽收，不可直接修改地址，应引导用户联系物流。",
            "询问已发货订单是否可修改地址",
            "direct_answer",
            False,
            [],
            ["已发货已揽收不可修改地址"],
            [],
            {
                "user_request": "修改地址",
                "shipping_status": "已发货",
                "logistics_status": "已揽收",
                "new_address_provided": "是",
                "order_id_provided": "是",
                "user_emotion": "焦虑",
            },
        )
        add_cases(
            bank,
            "train_change_unknown",
            "修改地址",
            CHANGE_UNKNOWN_TRAIN,
            "未提供当前订单阶段，先追问是否已发货及新地址信息。",
            "修改地址但状态信息不足",
            "ask_followup",
            False,
            ["shipping_status", "new_address"],
            ["修改地址状态未知先追问"],
            ["信息不足"],
            {
                "user_request": "修改地址",
                "shipping_status": "未知",
                "new_address_provided": "否",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_urge_no_order",
            "催发货",
            URGE_NO_ORDER_TRAIN,
            "用户焦虑催发货，但缺少订单定位信息。要先追问订单信息，并明确只能查询状态和反馈催促。",
            "催发货但未提供订单信息",
            "ask_followup",
            False,
            ["order_id"],
            ["催发货只能反馈不能承诺时间"],
            ["信息不足"],
            {
                "user_request": "催发货",
                "shipping_status": "未知",
                "order_id_provided": "否",
                "user_emotion": "焦虑",
            },
        )
        add_cases(
            bank,
            "train_urge_known",
            "催发货",
            URGE_KNOWN_TRAIN,
            "已知订单未发货，可安抚并说明会反馈催促，但不能承诺具体发货时间或保证已加急。",
            "催发货且订单未发货",
            "direct_answer",
            False,
            [],
            ["催发货只能反馈不能承诺时间"],
            [],
            {
                "user_request": "催发货",
                "shipping_status": "未发货",
                "order_id_provided": "是",
                "user_emotion": "焦虑",
            },
        )
        add_cases(
            bank,
            "train_urge_transfer",
            "催发货",
            URGE_TRANSFER_TRAIN,
            "用户强烈投诉且明确要求人工，需转人工，不要承诺发货时间。",
            "催发货并强烈投诉要求人工",
            "transfer_human",
            True,
            [],
            ["投诉高风险问题优先转人工", "催发货只能反馈不能承诺时间"],
            ["投诉风险", "情绪升级"],
            {
                "user_request": "催发货",
                "shipping_status": "未发货",
                "order_id_provided": "是",
                "user_emotion": "投诉倾向",
            },
            multi_turn=True,
        )
        add_cases(
            bank,
            "train_logistics_transit",
            "查询物流进度",
            LOGISTICS_TRANSIT_TRAIN,
            "物流运输中，可直接解释当前状态，不承诺送达时间。",
            "询问运输中物流为何未到",
            "direct_answer",
            False,
            [],
            ["物流运输中不承诺送达时间"],
            [],
            {
                "user_request": "查询物流",
                "shipping_status": "已发货",
                "logistics_status": "运输中",
                "is_signed": "否",
                "order_id_provided": "是",
                "user_emotion": "焦虑",
            },
        )
        add_cases(
            bank,
            "train_logistics_unknown",
            "查询物流进度",
            LOGISTICS_UNKNOWN_TRAIN,
            "缺少订单定位信息时，先追问订单号或手机号尾号，不能直接报物流节点。",
            "查询物流但未提供订单信息",
            "ask_followup",
            False,
            ["order_id"],
            ["查询物流缺少订单信息先追问"],
            ["信息不足"],
            {
                "user_request": "查询物流",
                "logistics_status": "未知",
                "order_id_provided": "否",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_logistics_stalled",
            "物流异常",
            LOGISTICS_STALLED_TRAIN,
            "物流停滞数天但用户只是焦虑，可解释并建议关注，不编造异常原因。",
            "物流停滞但未升级投诉",
            "direct_answer",
            False,
            [],
            ["物流异常不可编造原因"],
            [],
            {
                "user_request": "物流异常",
                "shipping_status": "已发货",
                "logistics_status": "运输中",
                "logistics_delay_days": 3,
                "order_id_provided": "是",
                "user_emotion": "焦虑",
            },
        )
        add_cases(
            bank,
            "train_logistics_transfer",
            "物流异常",
            LOGISTICS_STALLED_TRANSFER_TRAIN,
            "物流长时间停滞且用户投诉或怀疑丢件，应转人工。",
            "物流停滞并要求人工跟进",
            "transfer_human",
            True,
            [],
            ["物流异常投诉场景转人工"],
            ["投诉风险", "情绪升级"],
            {
                "user_request": "物流异常",
                "shipping_status": "已发货",
                "logistics_status": "运输中",
                "logistics_delay_days": 5,
                "order_id_provided": "是",
                "user_emotion": "投诉倾向",
            },
            multi_turn=True,
        )
        add_cases(
            bank,
            "train_refund_generic",
            "退款流程说明",
            REFUND_GENERIC_TRAIN,
            "用户只说退款，关键信息不足。需先追问是否已收货、是否需要退货、是否有质量问题。",
            "泛化退款咨询但信息不足",
            "ask_followup",
            False,
            ["is_signed", "has_quality_issue"],
            ["退款流程说明需先区分收货状态和质量问题"],
            ["信息不足"],
            {
                "user_request": "退款",
                "is_signed": "未知",
                "has_quality_issue": "未知",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_return_unopened",
            "退货条件判断",
            RETURN_UNOPENED_TRAIN,
            "商品已签收但未拆封且无质量问题，可进入退货退款判断，不要写成质量问题售后。",
            "询问未拆封商品是否可退货",
            "direct_answer",
            False,
            [],
            ["未拆封可进入退货退款判断"],
            [],
            {
                "user_request": "退货退款",
                "is_signed": "是",
                "is_opened": "否",
                "has_quality_issue": "否",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_return_opened_quality_yes",
            "退货条件判断",
            RETURN_OPENED_QUALITY_YES_TRAIN,
            "商品已拆封且有质量问题，可走质量问题售后，不能写成无理由退货。",
            "询问拆封且有质量问题商品是否可售后",
            "direct_answer",
            False,
            [],
            ["拆封后仅质量问题可售后"],
            [],
            {
                "user_request": "售后",
                "is_signed": "是",
                "is_opened": "是",
                "has_quality_issue": "是",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_return_opened_quality_no",
            "退货条件判断",
            RETURN_OPENED_QUALITY_NO_TRAIN,
            "商品已拆封且无质量问题，不支持售后，不要主动建议转人工。",
            "询问拆封无质量问题商品是否可退货",
            "direct_answer",
            False,
            [],
            ["拆封后无质量问题不支持售后"],
            [],
            {
                "user_request": "退货退款",
                "is_signed": "是",
                "is_opened": "是",
                "has_quality_issue": "否",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_return_missing",
            "退货条件判断",
            RETURN_MISSING_INFO_TRAIN,
            "退货条件判断信息不足时，先追问是否拆封及是否有质量问题。",
            "询问是否可退货但信息不足",
            "ask_followup",
            False,
            ["is_opened", "has_quality_issue"],
            ["退货条件判断需先确认是否拆封和是否质量问题"],
            ["信息不足"],
            {
                "user_request": "退货退款",
                "is_signed": "是",
                "is_opened": "未知",
                "has_quality_issue": "未知",
                "order_id_provided": "是",
                "user_emotion": "正常",
            },
        )
        add_cases(
            bank,
            "train_transfer_explicit",
            "转人工",
            TRANSFER_EXPLICIT_TRAIN,
            "用户明确要求人工客服时，应直接转人工并说明下一步。",
            "用户明确要求人工客服",
            "transfer_human",
            True,
            [],
            ["用户明确要求人工客服应转人工"],
            [],
            {
                "user_request": "转人工",
                "order_id_provided": "未知",
                "user_emotion": "正常",
            },
        )
        return bank

    add_cases(
        bank,
        "eval_cancel_unshipped",
        "取消订单",
        CANCEL_UNSHIPPED_EVAL,
        "未发货订单取消，要求回答清晰直接并给路径。",
        "评测未发货订单取消",
        "direct_answer",
        False,
        [],
        ["未发货可取消"],
        [],
        {
            "user_request": "取消订单",
            "shipping_status": "未发货",
            "order_id_provided": "是",
            "user_emotion": "正常",
        },
    )
    add_cases(
        bank,
        "eval_cancel_shipped",
        "取消订单",
        CANCEL_SHIPPED_EVAL,
        "已发货订单取消，要求明确说明不可直接取消并引导退货。",
        "评测已发货订单取消",
        "direct_answer",
        False,
        [],
        ["已发货不可取消应引导退货"],
        [],
        {
            "user_request": "取消订单",
            "shipping_status": "已发货",
            "logistics_status": "运输中",
            "is_signed": "否",
            "order_id_provided": "是",
            "user_emotion": "正常",
        },
    )
    add_cases(
        bank,
        "eval_change_unshipped",
        "修改地址",
        CHANGE_UNSHIPPED_EVAL,
        "未发货订单改地址，回答应直接、稳健。",
        "评测未发货改地址",
        "direct_answer",
        False,
        [],
        ["未发货可修改地址"],
        [],
        {
            "user_request": "修改地址",
            "shipping_status": "未发货",
            "new_address_provided": "是",
            "order_id_provided": "是",
            "user_emotion": "正常",
        },
    )
    add_cases(
        bank,
        "eval_change_collected",
        "修改地址",
        CHANGE_COLLECTED_EVAL,
        "已揽收订单改地址，要求明确引导联系物流。",
        "评测已揽收改地址",
        "direct_answer",
        False,
        [],
        ["已发货已揽收不可修改地址"],
        [],
        {
            "user_request": "修改地址",
            "shipping_status": "已发货",
            "logistics_status": "已揽收",
            "new_address_provided": "是",
            "order_id_provided": "是",
            "user_emotion": "焦虑",
        },
    )
    add_cases(
        bank,
        "eval_urge_no_order",
        "催发货",
        URGE_NO_ORDER_EVAL,
        "缺少订单信息的催发货场景，应先追问，再说明只能反馈催促。",
        "评测催发货但缺少订单信息",
        "ask_followup",
        False,
        ["order_id"],
        ["催发货只能反馈不能承诺时间"],
        ["信息不足"],
        {
            "user_request": "催发货",
            "shipping_status": "未知",
            "order_id_provided": "否",
            "user_emotion": "焦虑",
        },
    )
    add_cases(
        bank,
        "eval_logistics_transit",
        "查询物流进度",
        LOGISTICS_TRANSIT_EVAL,
        "运输中物流查询，不承诺送达时间。",
        "评测运输中物流查询",
        "direct_answer",
        False,
        [],
        ["物流运输中不承诺送达时间"],
        [],
        {
            "user_request": "查询物流",
            "shipping_status": "已发货",
            "logistics_status": "运输中",
            "is_signed": "否",
            "order_id_provided": "是",
            "user_emotion": "焦虑",
        },
    )
    add_cases(
        bank,
        "eval_refund_generic",
        "退款流程说明",
        REFUND_GENERIC_EVAL,
        "退款流程说明信息不足时，要追问已收货状态和质量问题。",
        "评测泛化退款咨询",
        "ask_followup",
        False,
        ["is_signed", "has_quality_issue"],
        ["退款流程说明需先区分收货状态和质量问题"],
        ["信息不足"],
        {
            "user_request": "退款",
            "is_signed": "未知",
            "has_quality_issue": "未知",
            "order_id_provided": "是",
            "user_emotion": "正常",
        },
    )
    add_cases(
        bank,
        "eval_return_missing",
        "退货条件判断",
        [
            "这个商品我想退掉，帮我判断下",
            "我想申请退货，你先帮我看看符不符合条件",
            "这个东西能不能退？",
            "帮我判断一下现在还能不能退货",
        ],
        "退货条件判断信息不足，要先追问拆封与质量问题。",
        "评测退货条件判断信息不足",
        "ask_followup",
        False,
        ["is_opened", "has_quality_issue"],
        ["退货条件判断需先确认是否拆封和是否质量问题"],
        ["信息不足"],
        {
            "user_request": "退货退款",
            "is_signed": "是",
            "is_opened": "未知",
            "has_quality_issue": "未知",
            "order_id_provided": "是",
            "user_emotion": "正常",
        },
    )
    return bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled golden train/eval datasets with DeepSeek."
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="train",
        help="Dataset split to generate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Accepted output JSONL path. Defaults depend on split.",
    )
    parser.add_argument(
        "--rejected-output",
        default=None,
        help="Rejected output JSONL path. Defaults depend on split.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="How many cases to generate from the selected split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset in the case bank, useful for batched runs.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Sleep time between API calls.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        help="DeepSeek model name.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help="DeepSeek API base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.55,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files instead of appending.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def read_doc(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required doc not found: {path}")
    return path.read_text(encoding="utf-8")


def get_api_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("Missing API key. Please set DEEPSEEK_API_KEY in .env or the environment.")


def build_messages(business_doc: str, io_doc: str, schema_doc: str, case: CaseSpec) -> list[dict[str, str]]:
    system_content = (
        "你是资深中文电商客服训练数据设计师。"
        "你只需要根据给定 case facts 写出一条高质量客服回复，并补充一个简短 intent。"
        "只能输出一个 JSON 对象，且仅包含 intent 和 target 两个字段。"
        "不要改写 case facts，不要输出 Markdown，不要输出解释。"
        "必须遵守："
        "1. target 只能是当前轮客服回复；"
        "2. 不编造订单状态、物流进度、处理结果、承诺时间；"
        "3. 信息不足时必须追问最少必要信息；"
        "4. should_transfer=true 时要明确转人工；"
        "5. should_transfer=false 时不要把转人工作为主要方案；"
        "6. order_id_provided=是时，不要再次索要订单号；"
        "7. 不要写‘根据平台规则’‘平台规定’；"
        "8. 回复控制在 1 到 3 句，简洁自然。"
    )
    user_content = (
        "请基于以下文档和 case facts 生成 intent 与 target。\n\n"
        "[业务分析文档]\n"
        f"{business_doc}\n\n"
        "[input/output 设计文档]\n"
        f"{io_doc}\n\n"
        "[schema 设计文档]\n"
        f"{schema_doc}\n\n"
        "[case facts]\n"
        f"scenario: {case.scenario}\n"
        f"instruction: {case.instruction}\n"
        f"intent anchor: {case.intent}\n"
        f"decision_type: {case.decision_type}\n"
        f"should_transfer: {json.dumps(case.should_transfer, ensure_ascii=False)}\n"
        f"missing_slots: {json.dumps(case.missing_slots, ensure_ascii=False)}\n"
        f"policy_basis: {json.dumps(case.policy_basis, ensure_ascii=False)}\n"
        f"risk_flags: {json.dumps(case.risk_flags, ensure_ascii=False)}\n"
        f"context: {json.dumps(case.context, ensure_ascii=False)}\n"
        f"history: {json.dumps(case.history, ensure_ascii=False)}\n\n"
        "只输出："
        '{"intent": "...", "target": "..."}'
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def call_chat_completions(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    parsed = json.loads(body)
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {body}") from exc


def build_sample(case: CaseSpec, model_output: dict[str, Any]) -> dict[str, Any]:
    intent = str(model_output.get("intent", case.intent)).strip() or case.intent
    target = str(model_output.get("target", "")).strip()
    return {
        "id": case.slug,
        "version": "schema_v1",
        "train_payload": {
            "system": SYSTEM_PROMPT,
            "context": dict(case.context),
            "history": list(case.history),
            "target": target,
        },
        "annotation_meta": {
            "scenario": case.scenario,
            "intent": intent,
            "decision_type": case.decision_type,
            "should_transfer": case.should_transfer,
            "missing_slots": list(case.missing_slots),
            "policy_basis": list(case.policy_basis),
            "risk_flags": list(case.risk_flags),
            "quality_check": {
                "is_consistent_with_context": True,
                "contains_forbidden_promise": False,
                "needs_revision": False,
            },
        },
    }


def contains_any(text: str, options: list[str]) -> bool:
    return any(option in text for option in options)


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    train_payload = sample.get("train_payload")
    annotation_meta = sample.get("annotation_meta")
    if not isinstance(train_payload, dict):
        return ["train_payload must be an object"]
    if not isinstance(annotation_meta, dict):
        return ["annotation_meta must be an object"]

    target = train_payload.get("target")
    context = train_payload.get("context")
    history = train_payload.get("history")
    if not isinstance(target, str) or not target.strip():
        errors.append("train_payload.target must be a non-empty string")
        return errors
    if not isinstance(context, dict):
        errors.append("train_payload.context must be an object")
        return errors
    if not isinstance(history, list) or not history:
        errors.append("train_payload.history must be a non-empty array")

    for phrase in FORBIDDEN_PROMISES:
        if phrase in target:
            errors.append(f"target contains forbidden promise: {phrase}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in target:
            errors.append(f"target contains discouraged unsupported phrase: {phrase}")

    decision_type = annotation_meta.get("decision_type")
    if decision_type not in DECISION_TYPES:
        errors.append("annotation_meta.decision_type is invalid")
    should_transfer = annotation_meta.get("should_transfer")
    if not isinstance(should_transfer, bool):
        errors.append("annotation_meta.should_transfer must be boolean")
    if decision_type == "ask_followup" and not annotation_meta.get("missing_slots"):
        errors.append("ask_followup samples must include missing_slots")

    if context.get("order_id_provided") == "是" and contains_any(target, ORDER_ID_REQUEST_HINTS):
        errors.append("target should not ask for order info again when order_id_provided is 是")

    if should_transfer and not contains_any(target, TRANSFER_HINTS):
        errors.append("transfer_human sample should mention人工")

    scenario = annotation_meta.get("scenario")
    shipping_status = context.get("shipping_status")
    logistics_status = context.get("logistics_status")
    is_opened = context.get("is_opened")
    has_quality_issue = context.get("has_quality_issue")

    if scenario == "取消订单" and shipping_status == "未发货":
        if not contains_any(target, ["可以取消", "可取消"]):
            errors.append("unshipped cancel sample should state it can be canceled")
    if scenario == "取消订单" and shipping_status == "已发货":
        if not contains_any(target, ["无法直接取消", "不能直接取消", "不可直接取消", "不支持直接取消"]):
            errors.append("shipped cancel sample should state it cannot be directly canceled")
        if not contains_any(target, ["退货", "退货退款"]):
            errors.append("shipped cancel sample should guide to return flow")
    if scenario == "修改地址" and (shipping_status == "已发货" or logistics_status == "已揽收"):
        if not contains_any(target, ["无法直接修改", "不能直接修改", "不可直接修改"]):
            errors.append("shipped address-change sample should state it cannot be directly changed")
        if "物流" not in target:
            errors.append("shipped address-change sample should guide the user to contact logistics")
    if scenario == "催发货":
        if contains_any(target, ["今天发", "马上发出", "一定发货"]):
            errors.append("urge-shipping sample must not promise shipment timing")
    if scenario == "退货条件判断" and is_opened == "是" and has_quality_issue == "否":
        if not contains_any(target, ["不支持售后", "不支持退货", "当前不支持", "无法办理"]):
            errors.append("opened non-quality return sample should clearly refuse after-sales")
    if scenario == "退货条件判断" and is_opened == "是" and has_quality_issue == "是":
        if not contains_any(target, ["质量问题售后", "售后", "退货退款"]):
            errors.append("opened quality-issue sample should mention after-sales path")

    return errors


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_parent_dir(path)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_output_paths(split: str) -> tuple[Path, Path]:
    accepted = ROOT / "data" / f"golden_v1_{split}.jsonl"
    rejected = ROOT / "data" / f"golden_v1_{split}_rejected.jsonl"
    return accepted, rejected


def select_cases(split: str, num_samples: int, start_index: int) -> list[CaseSpec]:
    bank = build_case_bank(split)
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    end_index = start_index + num_samples
    if end_index > len(bank):
        raise ValueError(
            f"Requested cases [{start_index}, {end_index}) exceed available {split} cases: {len(bank)}"
        )
    return bank[start_index:end_index]


def main() -> int:
    args = parse_args()
    load_env_file(ENV_PATH)

    business_doc = read_doc(DOC_DIR / "业务分析文档.markdown")
    io_doc = read_doc(DOC_DIR / "input_output设计v1.markdown")
    schema_doc = read_doc(DOC_DIR / "schema设计v1.markdown")
    api_key = get_api_key()

    selected_cases = select_cases(args.split, args.num_samples, args.start_index)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        print(f"[{index}/{len(selected_cases)}] Generating case={case.slug} ...", flush=True)
        try:
            content = call_chat_completions(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                messages=build_messages(business_doc, io_doc, schema_doc, case),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            parsed = json.loads(content)
            sample = build_sample(case, parsed)
            errors = validate_sample(sample)
        except Exception as exc:  # noqa: BLE001
            rejected_rows.append({"case_slug": case.slug, "error": f"request_or_parse_error: {exc}"})
            print(f"  rejected: {exc}", flush=True)
        else:
            if errors:
                rejected_rows.append(
                    {
                        "case_slug": case.slug,
                        "sample": sample,
                        "validation_errors": errors,
                    }
                )
                print(f"  rejected: {'; '.join(errors)}", flush=True)
            else:
                accepted_rows.append(sample)
                print("  accepted", flush=True)
        if index < len(selected_cases):
            time.sleep(args.sleep_seconds)

    output_path, rejected_path = default_output_paths(args.split)
    if args.output:
        output_path = Path(args.output)
    if args.rejected_output:
        rejected_path = Path(args.rejected_output)

    write_jsonl(output_path, accepted_rows, overwrite=args.overwrite)
    write_jsonl(rejected_path, rejected_rows, overwrite=args.overwrite)

    print("\nDone.")
    print(f"Split: {args.split}")
    print(f"Accepted: {len(accepted_rows)} -> {output_path}")
    print(f"Rejected: {len(rejected_rows)} -> {rejected_path}")
    print(f"Requested range: start={args.start_index}, count={args.num_samples}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
