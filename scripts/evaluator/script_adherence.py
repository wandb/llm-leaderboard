from collections import Counter
from typing import Any
import re

import numpy as np
import pandas as pd
import wandb

from config_singleton import WandbConfigSingleton
from utils import read_wandb_table


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SIMPLIFIED_ONLY_CHARS = set(
    ""
    "万与丑专业丛东丝丢两严丧个丰临为丽举么义乌乐乔习乡书买乱争于亏云亚产亩亲"
    "亵亿仅从仑仓仪们价众优伙会伞伟传伤伦伪体余佣佥侠侣侥侦侧侨侩侪侬俣俦俨"
    "俩俪俭债倾偬偻偾偿傥储儿兑党兰关兴养兽内冈册写军农冯冲决况冻净凄准凉减"
    "凑凛几凤凭凯击凿刍刘则刚创删别刬刭剂剐剑剧劝办务劢动励劲劳势勋匀区医华"
    "协单卖卢卤卫却厂厅历厉压厌厍厘厢厣厦厨厩县叁参双发变叙叠叶号叹叽吁后吓"
    "吕吗吨听启吴呒呓呕呖呗员呙呛呜咏咙咛响哑哒哓哔哕哗哙哜哝哟唠唢唤啧啬"
    "啭啮啰啸喷喽嗫嗳嘘嘤嘱噜嚣团园囱围囵国图圆圣圹场坏块坚坛坜坝坞坟坠垄"
    "垅垆垒垦垩垫垭垱垲垴埘埙埚埯堑堕墙壮声壳壶处备复够头夹夺奋奖奥妆妇妈"
    "妩妪妫姗娄娅娆娇娈娱娲娴婳婴婵婶媪嫒嫔嫱嬷孙学孪宁宝实宠审宪宫宽宾寝"
    "对寻导寿将尔尘尝尧尴尽层屉届属屡屦屿岁岂岖岗岘岙岚岛岭岳岽岿峡峣峤峥"
    "峦崂崃崄崭嵘嵚嵛嵝巅巩币帅师帏帐帘帜带帧帮帱帻帼幂庄庆庐庑库应庙庞废"
    "庼廪开异弃张弥弪弯弹强归当录彦彻径徕忆忏忧忾怀态怂怃怄怅怆怜总怼怿恋"
    "恒恳恶恸恹恺恻恼恽悦悬悭悯惊惧惨惩惫惬惭惮惯愠愤愦愿慑懑懒戆戋戏战"
    "户扑执扩扪扫扬扰抚抛抟抠抡抢护报担拟拢拣拥拦拧拨择挚挛挜挝挞挟挠挡"
    "挢挣挤挥挦捞损捡换捣据掳掴掷掸掺揽揿搀搁搂搅携摄摆摇摈摊撄撑撵撷撸"
    "撺擞攒敌敛数斋斓斗斩断无旧时旷旸昙昼显晋晒晓晔晕晖暂术机杀杂权杆条"
    "来杨极构枞枢枣枥枧枨枪枫枭柜柠柽栀栅标栈栉栋栌栎栏树栖样栾桠桡桢档"
    "桤桥桦桧桨桩梦检棂椁椟椠椤椭楼榄榅榇榈榉槚槛槟槠横樯樱橥橱橹橼檩欢"
    "欤欧歼殁殇残殒殓殚殡殴毁毕毙毡气氢氩氲汇汉汤汹沟没沣沤沥沦沧沪泞泪"
    "泶泷泸泺泻泼泽泾洁洒洼浃浅浆浇浈浊测浍济浏浐浑浒浓浔涂涛涝涞涟涠涡"
    "涢涣涤润涧涨涩淀渊渌渍渎渐渑渔渖渗温湾湿溃溅溆溇滗滚滞滟满滢滤滥滦"
    "滨滩滪漤潆潇潋潍潜潴澜濑濒灭灯灵灾灿炀炉炖炜炝点炼炽烁烂烃烛烟烦烧"
    "烨烩烫烬热焕焖爱爷牍牵牺犊状犷犸犹狈狝狞独狭狮狯狰狱狲猃猎猕猡猪猫"
    "猬献獭玑玛玮环现玺珐珑珲琏琐琼瑶瑷璎瓒电画畅畴疖疗疟疠疡疬疮疯疱痈"
    "痉痨痪痫瘅瘆瘘瘪瘫瘾瘿癞癣癫皑皱皲盏盐监盖盗盘眍眦睁睐睑瞒瞩矫矶矾"
    "矿砀码砖砗砚砜砺砻砾础硁硕硖硗硙确硷碍碛碜碱礼祢祯祷祸禅离秃秆种积"
    "称秽秾税稣稳穑穷窍窑窜窝窥窦竞笃笋笔笕笺笼笾筑筚筛筝筹签简箓箦箧箨"
    "箩箪箫篑篓篮篱簖籁籴类籼粜粝粤粪粮紧纠纡红纣纤纥约级纨纪纫纬纭纯纰"
    "纱纲纳纵纶纷纸纹纺纽纾线绀练组绅细织终绉绊绍绎经绑绒结绕绘给绚络绝"
    "绞统绢绣绥绦继绩绪绫续绮绯绰绲绳维绵绶绷绸绺综绽绾绿缀缁缄缅缆缇缈"
    "缉缋缌缍缎缓缔缕编缘缙缚缛缜缝缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰"
    "缱缴罂网罗罚罢罴羁羟翘耸耻聂聋职聍联聩聪肃肠肤肾肿胀胁胆胜胧胨胪胫"
    "胶脉脍脏脐脑脓脔脚脱脸腊腻腾膑臜舆舰舱艰艳艺节芜芦苁苇苈苋苌苍苏苹"
    "茎茏茑茔茕茧荆荐荚荛荜荞荟荠荡荣荤荥荦荧荨荩荪荫药莱莲莳莴获莹莺莼"
    "萝萤营萦萧萨葱蒋蓝蓟蓠蓣蓦蔷蔹蔺蔼蕴薮藓虑虚虫虬虮虽虾虿蚀蚁蚂蚕蛊"
    "蛎蛏蛮蛰蛱蛲蛳蛴蜕蜗蝇蝈蝉蝎蝼蝾螨衅衔补衬袄袅袭装裆裢裣裤裥褛褴"
    "见观规觅视觇览觉觊觋觌觎觏觐觑觞触觯誉誊计订讣认讥讦讧讨让讪讫训议"
    "讯记讲讳讴讵讶讷许讹论讼讽设访诀证诂评诅识诈诉诊诋词诎诏译试诗诚诛"
    "话诞诟诠诡询诣该详诧语误诰诱诲说诵请诸诺读课谁调谄谅谆谈谊谋谌谍谎"
    "谏谐谑谓谔谕谗谘谙谚谜谟谢谣谤谦谧谨谩谪谬谱谴谵谷贝贞负贡财责贤败"
    "账货质贩贪贫贬购贮贯贰贱贴贵贷贸费贺贻贼贾贿赁赂资赅赋赌赎赏赐赔赖"
    "赘赚赛赞赠赡赢赣赵赶趋趱跃跄践跷跸跹跻踊踌踪踬蹑蹒蹿躏躯车轧轨轩转"
    "轮软轰轱轲轴轵轻载轿较辅辆辈辉辊辍辑输辖辗辘辙辞辩辫边辽达迁过迈运"
    "还这进远违连迟适选逊递逻遗邓邮邻郑酝酱酿释鉴针钉钊钓钙钛钝钞钟钢钥"
    "钦钧钨钩钪钮钱钲铁铃铄铅铜铝铠铡铢铭铮银铺链销锁锂锅锋锌锐错锚锡锢"
    "锣锤锦键锯锰锵锶锷锻镀镁镇镜镝镞镟镣镭镰长门问闯闲间闵闷闸闹闻阅阔"
    "队阳阴阵阶际陆陈陕陨险随隐隶难雏雠雳雾霁霭静靥鞑鞒韦韩韵页顶顷项顺"
    "须顾顿颁颂预领颇颈颊频颗题额颚颜愿颠颤风飞饥饭饮饯饰饱饲饵饶馆馈馊"
    "馋馒马驭驮驯驰驱驳驶驷驻驼驾驿骂骄验骏骑骗骚骛骜骤骥髅髋鬓魇鱼鲁鲜"
    "鲤鲸鳄鳍鳖鳗鳞鸟鸡鸣鸭鸯鸳鹅鹉鹊鹏鹤鹦鹰麦黄黉黩齐齿龄龃龅龋龙龟"
)
AMBIGUOUS_FALLBACK_CHARS = set("后着斗余几干台面只系云于并复")
SIMPLIFIED_ONLY_CHARS -= AMBIGUOUS_FALLBACK_CHARS


def _get_opencc_converter() -> Any:
    try:
        from opencc import OpenCC
    except Exception:
        return None

    for config_name in ("s2twp", "s2tw", "s2t"):
        try:
            return OpenCC(config_name)
        except Exception:
            continue
    return None


def _is_han(char: str) -> bool:
    return bool(HAN_RE.fullmatch(char))


def _script_stats(text: Any, converter: Any = None) -> dict[str, Any]:
    text = "" if text is None else str(text)
    han_chars = [char for char in text if _is_han(char)]
    simplified_counter: Counter[str] = Counter()

    if converter is not None:
        converted = converter.convert(text)
        if len(converted) == len(text):
            for original, converted_char in zip(text, converted):
                if _is_han(original) and original != converted_char:
                    simplified_counter[original] += 1
        else:
            for char in han_chars:
                if converter.convert(char) != char:
                    simplified_counter[char] += 1
    else:
        simplified_counter.update(char for char in han_chars if char in SIMPLIFIED_ONLY_CHARS)

    han_count = len(han_chars)
    simplified_count = sum(simplified_counter.values())
    intrusion_rate = simplified_count / han_count if han_count else np.nan
    return {
        "han_char_count": han_count,
        "simplified_char_count": simplified_count,
        "simplified_intrusion_rate": intrusion_rate,
        "script_adherence_score": 1.0 - intrusion_rate if han_count else np.nan,
        "simplified_chars_sample": "".join(char for char, _ in simplified_counter.most_common(30)),
    }


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    script_cfg = cfg.get("script_adherence", {})
    source_table = script_cfg.get("source_table", "mtbench_output_table")
    answer_column = script_cfg.get("answer_column", "answer")
    min_han_chars = int(script_cfg.get("min_han_chars", 1))

    df = read_wandb_table(table_name=source_table, run=run)
    if answer_column not in df.columns:
        raise KeyError(f"{answer_column} not found in {source_table}")

    dedupe_cols = [
        col
        for col in ["model_name", "question_id", "turn", answer_column]
        if col in df.columns
    ]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols).copy()

    converter = _get_opencc_converter()
    detection_method = "opencc_s2t" if converter is not None else "curated_simplified_char_set"
    stat_rows = []
    for _, row in df.iterrows():
        stats = _script_stats(row[answer_column], converter=converter)
        stat_rows.append({**row.to_dict(), **stats, "detection_method": detection_method})

    output_df = pd.DataFrame(stat_rows)
    scored_df = output_df[output_df["han_char_count"] >= min_han_chars].copy()
    if scored_df.empty:
        leaderboard_df = pd.DataFrame(
            [
                {
                    "model_name": cfg.model.pretrained_model_name_or_path,
                    "script_adherence_score": np.nan,
                    "simplified_intrusion_rate": np.nan,
                    "answer_count": len(output_df),
                    "scored_answer_count": 0,
                    "han_char_count": 0,
                    "simplified_char_count": 0,
                    "detection_method": detection_method,
                    "source_table": source_table,
                }
            ]
        )
    else:
        model_col = "model_name" if "model_name" in scored_df.columns else None
        group_keys = [model_col] if model_col else []
        grouped = scored_df.groupby(group_keys, dropna=False) if group_keys else [(None, scored_df)]
        leaderboard_rows = []
        for key, group in grouped:
            han_count = int(group["han_char_count"].sum())
            simplified_count = int(group["simplified_char_count"].sum())
            intrusion_rate = simplified_count / han_count if han_count else np.nan
            model_name = key if model_col else cfg.model.pretrained_model_name_or_path
            if isinstance(model_name, tuple):
                model_name = model_name[0]
            leaderboard_rows.append(
                {
                    "model_name": model_name,
                    "script_adherence_score": 1.0 - intrusion_rate if han_count else np.nan,
                    "simplified_intrusion_rate": intrusion_rate,
                    "answer_count": len(output_df),
                    "scored_answer_count": len(group),
                    "han_char_count": han_count,
                    "simplified_char_count": simplified_count,
                    "detection_method": detection_method,
                    "source_table": source_table,
                }
            )
        leaderboard_df = pd.DataFrame(leaderboard_rows)

    run.log(
        {
            "traditional_chinese_script_adherence_output_table": wandb.Table(dataframe=output_df),
            "traditional_chinese_script_adherence_leaderboard_table": wandb.Table(dataframe=leaderboard_df),
            "traditional_chinese_script_adherence_score": float(
                leaderboard_df["script_adherence_score"].iloc[0]
            ),
            "traditional_chinese_simplified_intrusion_rate": float(
                leaderboard_df["simplified_intrusion_rate"].iloc[0]
            ),
        }
    )
