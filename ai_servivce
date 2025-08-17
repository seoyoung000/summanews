# 📦 AI 모델
!pip install flask flask-cors pyngrok requests transformers torch -q

# 📚 모든 라이브러리 import
import os
import requests
import json
from datetime import datetime
import re
from typing import List, Dict
from dataclasses import dataclass
import warnings
import hashlib
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import random
warnings.filterwarnings('ignore')

# 🤖 AI 모델 import
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # max_length 경고 제거

os.environ['NAVER_CLIENT_ID'] = 'fe9DLGhYbEVLy4sdQnVk'
os.environ['NAVER_CLIENT_SECRET'] = '2f0NEntTNN'
os.environ['NEWSAPI_KEY'] = 'a80b5826f01349c5824f4298d8f61eef'

print("✅ 설정 완료! (ai_service.py)")

# 📄 뉴스 기사 데이터 클래스
@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    published_at: str
    source: str
    category: str

# 🗞️ 뉴스 API 클래스
class SimpleNewsAPI:
    def __init__(self):
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')

        self.category_keywords = {
            '정치': ['정치', '국정감사', '대통령', '국회'],
            '경제': ['경제', '주식', '부동산', '금리'],
            '연예': ['연예', '드라마', 'K-pop', '영화'],
            '스포츠': ['스포츠', '축구', '야구', '올림픽'],
            "생활문화": ["생활", "문화", "취미", "공연"],
            "건강": ["건강", "질병", "의료", "운동"],
            "사회": ["사회", "사건사고", "교육", "노동"]
        }

    def get_naver_news(self, category: str, count: int = 5) -> List[NewsArticle]:
        try:
            keywords = self.category_keywords.get(category, [category])
            query = keywords[0]

            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {
                "X-Naver-Client-Id": self.naver_client_id,
                "X-Naver-Client-Secret": self.naver_client_secret
            }
            params = {
                "query": query,
                "display": count,
                "start": 1,
                "sort": "date"
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                articles = []
                for item in data['items']:
                    title = re.sub('<[^<]+?>', '', item['title'])
                    description = re.sub('<[^<]+?>', '', item['description'])

                    article = NewsArticle(
                        title=title,
                        content=description,
                        url=item['link'],
                        published_at=item['pubDate'],
                        source="네이버뉴스",
                        category=category
                    )
                    articles.append(article)
                return articles
            else:
                print(f"API 오류: {response.status_code}")
                return []

        except Exception as e:
            print(f"뉴스 가져오기 오류: {e}")
            return []

    def get_recommended_news(self, total_count: int = 5) -> List[NewsArticle]:
        """전체 장르에서 랜덤하게 뉴스를 가져오는 함수"""
        try:
            print(f"🎯 오늘의 추천 뉴스 {total_count}개 수집 중...")

            # 모든 카테고리에서 뉴스 수집
            all_articles = []
            categories = list(self.category_keywords.keys())

            # 각 카테고리에서 3개씩 가져와서 풀을 만듦
            for category in categories:
                try:
                    articles = self.get_naver_news(category, 3)
                    for article in articles:
                        article.category = category  # 카테고리 정보 확실히 설정
                    all_articles.extend(articles)
                    print(f"📰 {category} 카테고리: {len(articles)}개 수집")
                except Exception as e:
                    print(f"⚠️ {category} 카테고리 수집 실패: {e}")
                    continue

            if not all_articles:
                print("❌ 수집된 뉴스가 없습니다.")
                return []

            # 중복 제거 (제목 기준)
            unique_articles = []
            seen_titles = set()
            for article in all_articles:
                if article.title not in seen_titles:
                    unique_articles.append(article)
                    seen_titles.add(article.title)

            print(f"🔄 중복 제거 후: {len(unique_articles)}개 뉴스")

            # 랜덤하게 선택
            if len(unique_articles) >= total_count:
                selected_articles = random.sample(unique_articles, total_count)
            else:
                selected_articles = unique_articles

            # 카테고리별로 섞기
            random.shuffle(selected_articles)

            print(f"✅ 추천 뉴스 {len(selected_articles)}개 선택 완료!")
            for i, article in enumerate(selected_articles, 1):
                print(f"   {i}. [{article.category}] {article.title[:50]}...")

            return selected_articles

        except Exception as e:
            print(f"❌ 추천 뉴스 수집 오류: {e}")
            return []

# 🤖 AI 요약 클래스 (길이 최적화 버전)
class AISummarizer:
    def __init__(self):
        print("🤖 AI 요약 모델 로딩 중...")
        try:
            model_name = "eenzeenee/t5-base-korean-summarization"
            self.summarizer = pipeline(
                "summarization",
                model_name,
                tokenizer=model_name,
                device=-1,
                batch_size=4  # 배치 크기 설정
            )
            print("✅ AI 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ AI 모델 로딩 실패: {e}")
            self.summarizer = None

    def summarize_batch(self, texts, min_length=20, max_length=100):
        """여러 텍스트를 한 번에 배치 처리 (길이 조정)"""
        if self.summarizer is None:
            return [self.simple_summarize_fast(text, max_length) for text in texts]

        try:
            # 텍스트 전처리
            processed_texts = []
            for text in texts:
                if len(text) > 3000:  # 길이 제한 증가
                    text = text[:3000]
                processed_texts.append(text)

            # 배치로 한 번에 요약
            summaries = self.summarizer(
                processed_texts,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
                batch_size=len(texts)
            )

            # 자연스러운 문장 마무리 처리
            result_summaries = []
            for summary in summaries:
                text = summary['summary_text'].strip()
                text = self.fix_sentence_ending(text)
                result_summaries.append(text)

            return result_summaries

        except Exception as e:
            print(f"배치 요약 실패: {e}")
            return [self.simple_summarize_fast(text, max_length) for text in texts]

    def fix_sentence_ending(self, text: str) -> str:
        """문장 끝맺음을 자연스럽게 처리"""
        text = text.strip()

        # 이미 적절한 문장부호로 끝나면 그대로 반환
        if text.endswith(('.', '!', '?', '다', '음', '함', '됨', '임')):
            return text

        # 마지막 불완전한 단어나 구문 제거
        sentences = text.split('.')
        if len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            if len(last_sentence) < 10 or not any(char in last_sentence for char in '다음함됨임'):
                text = '.'.join(sentences[:-1]) + '.'
            else:
                text = text + '.' if not text.endswith('.') else text
        else:
            text = text + '.' if not text.endswith('.') else text

        return text

    def simple_summarize_fast(self, text: str, max_length: int = 100) -> str:
        """개선된 간단 요약 (길이별 최적화)"""
        import re

        # 더 정확한 문장 분리
        sentence_endings = re.split(r'[.!?]\s+', text)
        sentences = []

        for sent in sentence_endings:
            sent = sent.strip()
            # 의미있는 문장만 선택
            if len(sent) > 10 and not sent.isdigit() and '...' not in sent:
                sentences.append(sent)

        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text

        # 길이에 따른 요약 처리
        if max_length <= 60:  # 미리보기 (1줄, 매우 짧게)
            # 가장 핵심적인 문장 1개만 선택
            if len(sentences) >= 1:
                # 가장 길고 정보가 많은 문장 선택
                best_sentence = max(sentences[:3], key=len)
                # 너무 길면 앞부분만 잘라서 사용
                if len(best_sentence) > 55:
                    best_sentence = best_sentence[:50] + "..."
                return best_sentence + ("." if not best_sentence.endswith(('.', '!', '?')) else "")
            else:
                return text[:50] + "..."

        else:  # 상세 요약 (5문장 정도)
            # 더 많은 문장 포함하여 상세하게
            selected_sentences = sentences[:8]  # 최대 8개에서 선택

            # 중복 제거 및 품질 좋은 문장들 선택
            unique_sentences = []
            for sent in selected_sentences:
                # 더 까다로운 조건으로 좋은 문장만 선택
                if (sent not in unique_sentences and
                    len(sent) > 20 and  # 최소 길이 증가
                    not any(skip in sent for skip in ['...', '등등', '기타']) and
                    any(char in sent for char in '다음함됨임음')):  # 한국어 문장 확인
                    unique_sentences.append(sent)

            # 5문장 정도로 구성
            final_sentences = unique_sentences[:5]

            # 문장이 부족하면 원본에서 더 가져오기
            if len(final_sentences) < 3 and len(sentences) > len(final_sentences):
                additional = [s for s in sentences if s not in final_sentences and len(s) > 15]
                needed = min(2, len(additional))
                final_sentences.extend(additional[:needed])

            # 자연스러운 연결
            if final_sentences:
                result = ". ".join(final_sentences) + "."
                return result
            else:
                return text[:200] + "..."

    def summarize_articles_fast(self, articles: List[NewsArticle]) -> List[Dict]:
        """고속 병렬 요약 (길이 최적화)"""
        print(f"⚡ 고속 배치 요약 시작... ({len(articles)}개 기사)")

        # 모든 텍스트를 미리 준비
        full_texts = [f"{article.title}. {article.content}" for article in articles]

        # 미리보기용과 상세용을 병렬로 배치 처리 (길이 대폭 조정)
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 미리보기: 1줄 (15-60자) - 매우 짧게
            preview_future = executor.submit(
                self.summarize_batch, full_texts, 15, 60
            )
            # 상세 요약: 5문장 정도 (200-400자) - 더 자세하게
            detailed_future = executor.submit(
                self.summarize_batch, full_texts, 200, 400
            )

            # 결과 대기
            preview_summaries = preview_future.result()
            detailed_summaries = detailed_future.result()

        # 결과 조합
        summarized_articles = []
        for i, article in enumerate(articles):
            summarized_article = {
                "original_title": article.title,
                "preview_summary": preview_summaries[i],
                "detailed_summary": detailed_summaries[i],
                "url": article.url,
                "published_at": article.published_at,
                "source": article.source,
                "category": article.category
            }
            summarized_articles.append(summarized_article)

        print("✅ 고속 요약 완료!")
        return summarized_articles

class AINewsApp:
    def __init__(self):
        self.news_api = SimpleNewsAPI()
        self.summarizer = AISummarizer()
        self.cache = {}  # 메모리 캐시

    def get_cache_key(self, category, count):
        """캐시 키 생성"""
        return f"{category}_{count}_{datetime.now().strftime('%H')}"  # 1시간마다 갱신

    def get_dual_summarized_news_fast(self, category: str, count: int = 5) -> Dict:
        """캐시 + 고속 요약"""
        # 캐시 확인
        cache_key = self.get_cache_key(category, count)
        if cache_key in self.cache:
            print(f"📦 캐시에서 {category} 뉴스 반환")
            cached_result = self.cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        try:
            print(f"📰 {category} 카테고리 뉴스 가져오는 중...")
            articles = self.news_api.get_naver_news(category, count)

            if not articles:
                return {
                    "status": "error",
                    "message": "뉴스를 가져올 수 없습니다.",
                    "data": []
                }

            # 고속 요약 사용
            summarized_articles = self.summarizer.summarize_articles_fast(articles)

            result = {
                "status": "success",
                "category": category,
                "count": len(summarized_articles),
                "data": summarized_articles,
                "timestamp": datetime.now().isoformat(),
                "ai_enabled": self.summarizer.summarizer is not None,
                "cached": False
            }

            # 캐시 저장
            self.cache[cache_key] = result.copy()

            return result

        except Exception as e:
            print(f"❌ 뉴스 처리 오류: {e}")
            return {
                "status": "error",
                "message": f"오류 발생: {str(e)}",
                "data": []
            }

    def get_recommended_news_fast(self, count: int = 5) -> Dict:
        """오늘의 추천 뉴스 (전체 장르에서 랜덤 선택)"""
        # 추천 뉴스용 캐시 키 (1분마다 갱신)
        cache_key = f"recommended_{count}_{datetime.now().strftime('%H_%M')}"

        if cache_key in self.cache:
            print(f"📦 캐시에서 추천 뉴스 반환")
            cached_result = self.cache[cache_key].copy()
            cached_result["cached"] = True
            return cached_result

        try:
            print(f"🎯 오늘의 추천 뉴스 {count}개 생성 중...")

            # 전체 장르에서 랜덤하게 뉴스 가져오기
            articles = self.news_api.get_recommended_news(count)

            if not articles:
                return {
                    "status": "error",
                    "message": "추천 뉴스를 가져올 수 없습니다.",
                    "data": []
                }

            # 고속 요약 사용
            summarized_articles = self.summarizer.summarize_articles_fast(articles)

            result = {
                "status": "success",
                "category": "추천",
                "type": "recommended",
                "count": len(summarized_articles),
                "data": summarized_articles,
                "timestamp": datetime.now().isoformat(),
                "ai_enabled": self.summarizer.summarizer is not None,
                "cached": False,
                "description": "전체 장르에서 엄선한 오늘의 추천 뉴스"
            }

            # 캐시 저장 (1분 유지)
            self.cache[cache_key] = result.copy()

            return result

        except Exception as e:
            print(f"❌ 추천 뉴스 처리 오류: {e}")
            return {
                "status": "error",
                "message": f"오류 발생: {str(e)}",
                "data": []
            }

# 📄 web_server.py (최적화된 전체 코드)

import pytz
from flask import Flask, jsonify, request
from flask_cors import CORS
from pyngrok import ngrok

# Flask 앱 생성
app = Flask(__name__)
CORS(app)

print("🌐 AI 뉴스 웹서버 설정 중... (web_server.py)")

# 🗂️ 지원하는 뉴스 카테고리 정의
SUPPORTED_CATEGORIES = {
    '정치': {'icon': '🏛️', 'description': '정치, 정부, 국정감사 관련 뉴스'},
    '경제': {'icon': '💰', 'description': '주식, 부동산, 금리, 경제 정책 뉴스'},
    '연예': {'icon': '🎬', 'description': '드라마, K-pop, 영화, 연예인 뉴스'},
    '스포츠': {'icon': '⚽', 'description': '축구, 야구, 올림픽, 스포츠 경기 뉴스'},
    '사회': {'icon': '🏢', 'description': '사건사고, 교육, 노동, 사회 이슈 뉴스'},
    '건강': {'icon': '💊', 'description': '질병, 의료, 운동, 건강 관리 뉴스'},
    '생활문화': {'icon': '🎨', 'description': '취미, 공연, 문화, 라이프스타일 뉴스'}
}

# 영문 카테고리 매핑
CATEGORY_MAPPING = {
    'politics': '정치',
    'economy': '경제',
    'entertainment': '연예',
    'sports': '스포츠',
    'society': '사회',
    'health': '건강',
    'lifestyle': '생활문화',
    'recommended': '추천'  # 추천 뉴스 추가
}

ai_app = AINewsApp()
KST = pytz.timezone('Asia/Seoul')

# ngrok 경고 페이지 우회 헤더 추가
@app.after_request
def after_request(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/')
def home():
    """메인 페이지"""
    current_kst_time = datetime.now(KST)

    # 동적으로 API 링크 생성 (영문 URL 사용)
    api_links_html = ""
    for eng_category, kor_category in CATEGORY_MAPPING.items():
        if eng_category == 'recommended':
            api_links_html += f'<a href="/api/news/{eng_category}" class="api-link recommended" target="_blank">🎯 오늘의 추천 뉴스</a>'
        else:
            info = SUPPORTED_CATEGORIES[kor_category]
            api_links_html += f'<a href="/api/news/{eng_category}" class="api-link" target="_blank">{info["icon"]} {kor_category} 뉴스</a>'

    # 동적으로 테스트 버튼 생성
    test_buttons_html = ""
    test_buttons_html += '<button class="test-btn recommended" onclick="testAPI(\'recommended\')">🎯 오늘의 추천</button>'
    for eng_category, kor_category in CATEGORY_MAPPING.items():
        if eng_category != 'recommended':
            info = SUPPORTED_CATEGORIES[kor_category]
            test_buttons_html += f'<button class="test-btn" onclick="testAPI(\'{eng_category}\')">{info["icon"]} {kor_category} 뉴스</button>'

    test_buttons_html += '<button class="test-btn" onclick="testHealthAPI()">💚 서버 상태</button>'

    return f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🤖 AI 뉴스 요약 서버</title>
        <style>
            body {{
                font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                line-height: 1.6;
            }}
            .container {{
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(15px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid rgba(255,255,255,0.2);
            }}
            .status-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .status-card {{
                background: rgba(255,255,255,0.15);
                padding: 20px;
                border-radius: 15px;
                border-left: 5px solid #4CAF50;
                transition: transform 0.3s ease;
            }}
            .status-card:hover {{ transform: translateY(-5px); }}
            .api-section {{
                background: rgba(0,0,0,0.2);
                padding: 25px;
                border-radius: 15px;
                margin: 25px 0;
            }}
            .api-links {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .api-link {{
                display: block;
                background: rgba(255,255,255,0.2);
                padding: 15px;
                border-radius: 12px;
                text-decoration: none;
                color: white;
                transition: all 0.3s ease;
                text-align: center;
                font-weight: 500;
                font-size: 14px;
            }}
            .api-link:hover {{
                background: rgba(255,255,255,0.3);
                transform: scale(1.05);
            }}
            .api-link.recommended {{
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                font-weight: bold;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .api-link.recommended:hover {{
                background: linear-gradient(45deg, #FF5252, #26C6DA);
                transform: scale(1.08);
            }}
            .test-section {{
                background: linear-gradient(45deg, rgba(76,175,80,0.3), rgba(33,150,243,0.3));
                padding: 25px;
                border-radius: 15px;
                margin: 25px 0;
            }}
            .test-buttons {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 10px;
                margin-top: 20px;
            }}
            .test-btn {{
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                border: none;
                padding: 12px 16px;
                border-radius: 25px;
                color: white;
                font-weight: bold;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .test-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .test-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
            }}
            .test-btn.recommended {{
                background: linear-gradient(45deg, #FFD700, #FF8C00);
                font-size: 14px;
                font-weight: 900;
                box-shadow: 0 8px 20px rgba(255, 215, 0, 0.3);
            }}
            .test-btn.recommended:hover {{
                background: linear-gradient(45deg, #FFC107, #FF9800);
                transform: scale(1.1);
                box-shadow: 0 10px 25px rgba(255, 215, 0, 0.4);
            }}
            h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
            h2 {{ color: #FFE082; margin-top: 30px; }}
            h3 {{ color: #FFCC80; }}
            #test-result {{
                margin-top: 20px;
                display: none;
            }}
            .result-box {{
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 12px;
                max-height: 500px;
                overflow-y: auto;
            }}
            .news-article-item {{
                background: rgba(0,0,0,0.3);
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 3px solid #4ECDC4;
            }}
            .news-article-item:last-child {{
                margin-bottom: 0;
            }}
            .news-article-item.recommended {{
                border-left: 3px solid #FFD700;
                background: rgba(255, 215, 0, 0.1);
            }}
            .summary-preview {{
                background: rgba(76, 175, 80, 0.2);
                padding: 8px 12px;
                border-radius: 6px;
                margin: 6px 0;
                border-left: 3px solid #4CAF50;
                line-height: 1.4;
                font-size: 14px;
                font-weight: 500;
            }}
            .summary-detailed {{
                background: rgba(33, 150, 243, 0.2);
                padding: 12px 15px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 3px solid #2196F3;
                line-height: 1.6;
                font-size: 14px;
                max-height: 250px;
                overflow-y: auto;
            }}
            .speed-indicator {{
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 12px;
                margin-left: 10px;
            }}
            .recommendation-badge {{
                background: linear-gradient(45deg, #FFD700, #FF8C00);
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
                box-shadow: 0 2px 5px rgba(255, 215, 0, 0.3);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI 뉴스 요약 서버</h1>
                <p style="font-size: 1.2em; opacity: 0.9;">
                    실시간 AI 뉴스 요약 API 서비스 + 오늘의 추천 뉴스
                </p>
            </div>
            <div class="status-grid">
                <div class="status-card">
                    <h3>🚀 서버 상태</h3>
                    <p><strong>상태:</strong> 정상 작동 중</p>
                    <p><strong>시간:</strong> {current_kst_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="status-card">
                    <h3>🤖 AI 모델</h3>
                    <p><strong>모델:</strong> T5 한국어 요약 ✅</p>
                    <p><strong>요약 모드:</strong> 고속 배치 처리</p>
                </div>
                <div class="status-card">
                    <h3>🎯 새로운 기능</h3>
                    <p><strong>추천 뉴스:</strong> 전체 장르 랜덤 ✨</p>
                    <p><strong>업데이트:</strong> 1분마다 자동 갱신</p>
                </div>
            </div>
            <div class="api-section">
                <h2>🔗 API 엔드포인트</h2>
                <div class="api-links">
                    <a href="/api/test" class="api-link" target="_blank">✅ 기본 테스트</a>
                    <a href="/api/health" class="api-link" target="_blank">💚 상태 확인</a>
                    {api_links_html}
                </div>
            </div>
            <div class="api-section">
                <h2>📱 안드로이드 앱 연동</h2>
                <p><strong>Base URL:</strong> {request.host_url}</p>
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 13px; margin-top: 15px;">
// API 응답 구조<br>
{{<br>
&nbsp;&nbsp;"status": "success",<br>
&nbsp;&nbsp;"type": "recommended", // 추천 뉴스일 경우<br>
&nbsp;&nbsp;"cached": false,<br>
&nbsp;&nbsp;"data": [{{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"original_title": "뉴스 제목",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"preview_summary": "짧은 미리보기 요약",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"detailed_summary": "상세한 긴 요약",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"category": "정치", // 해당 뉴스의 카테고리<br>
&nbsp;&nbsp;&nbsp;&nbsp;"url": "원본 링크"<br>
&nbsp;&nbsp;}}]<br>
}}<br><br>
// 새로운 추천 뉴스 API<br>
@GET("api/news/recommended")<br>
Call&lt;NewsResponse&gt; getRecommendedNews();
                </div>
            </div>
            <div class="test-section">
                <h2>🧪 실시간 API 테스트 <span class="speed-indicator">길이 최적화</span> <span class="recommendation-badge">추천 기능 NEW!</span></h2>
                <div class="test-buttons">
                    {test_buttons_html}
                </div>
                <div id="test-result">
                    <div class="result-box">
                        <div id="test-content">테스트 결과가 여기에 표시됩니다...</div>
                    </div>
                </div>
            </div>
        </div>
        <script>
        const categoryNames = {{
            'politics': '정치',
            'economy': '경제',
            'entertainment': '연예',
            'sports': '스포츠',
            'society': '사회',
            'health': '건강',
            'lifestyle': '생활문화',
            'recommended': '오늘의 추천'
        }};

        let isRequesting = false;

        async function testAPI(category) {{
            if (isRequesting) {{
                console.log('이미 요청 중입니다...');
                return;
            }}

            isRequesting = true;

            const resultDiv = document.getElementById('test-result');
            const contentDiv = document.getElementById('test-content');
            const koreanName = categoryNames[category] || category;

            const buttons = document.querySelectorAll('.test-btn');
            buttons.forEach(btn => btn.disabled = true);

            resultDiv.style.display = 'block';

            let loadingDots = 0;
            const loadingInterval = setInterval(() => {{
                const dots = '.'.repeat((loadingDots % 3) + 1);
                if (category === 'recommended') {{
                    contentDiv.innerHTML = `🎯 전체 장르에서 추천 뉴스 수집 중${{dots}} (AI 처리 중...)`;
                }} else {{
                    contentDiv.innerHTML = `⚡ ${{koreanName}} 뉴스 고속 처리 중${{dots}} (평균 10-15초)`;
                }}
                loadingDots++;
            }}, 500);

            try {{
                const startTime = Date.now();
                const response = await fetch(`/api/news/${{category}}`, {{
                    headers: {{
                        'ngrok-skip-browser-warning': 'true'
                    }}
                }});
                const data = await response.json();
                const endTime = Date.now();

                clearInterval(loadingInterval);

                if (data.status === 'success') {{
                    const processingTime = ((endTime - startTime) / 1000).toFixed(1);
                    const cacheStatus = data.cached ? ' (캐시됨 ⚡)' : ' (AI 처리됨 🤖)';
                    const isRecommended = category === 'recommended';

                    let newsHtml = `
                        <h4 style="color: #4CAF50;">✅ ${{koreanName}} 뉴스 완료!
(${{processingTime}}초${{cacheStatus}})</h4>
                        <p><strong>📊 뉴스 개수:</strong> ${{data.count}}개</p>
                        <p><strong>🤖 AI 상태:</strong> ${{data.ai_enabled ? '✅ 활성화' : '❌ 비활성화'}}</p>
                    `;

                    if (isRecommended) {{
                        newsHtml += `<p><strong>🎯 추천 방식:</strong> ${{data.description || '전체 장르 랜덤 선택'}}</p>`;
                    }}

                    newsHtml += `
                        <hr style="border: 1px solid rgba(255,255,255,0.2); margin: 15px 0;">
                        <h5>📰 요약된 뉴스 기사:</h5>
                    `;

                    if (data.data && data.data.length > 0) {{
                        data.data.forEach((newsItem, index) => {{
                            const categoryBadge = isRecommended ? `<span style="background: linear-gradient(45deg, #FFD700, #FF8C00); padding: 2px 6px; border-radius: 8px; font-size: 10px; margin-left: 5px;">[${{newsItem.category}}]</span>` : '';

                            newsHtml += `
                                <div class="news-article-item ${{isRecommended ? 'recommended' : ''}}">
                                    <p><strong>${{index + 1}}. ${{newsItem.original_title || 'N/A'}} ${{categoryBadge}}</strong></p>
                                    <div class="summary-preview">
                                        <strong>📱 미리보기 요약:</strong><br>
                                        ${{newsItem.preview_summary || 'N/A'}}
                                    </div>
                                    <div class="summary-detailed">
                                        <strong>📖 상세 요약:</strong><br>
                                        ${{newsItem.detailed_summary || 'N/A'}}
                                    </div>
                                    <p><a href="${{newsItem.url}}" target="_blank" style="color:#81D4FA; text-decoration:none;">🔗 원본 뉴스 보기</a></p>
                                </div>
                            `;
                        }});
                    }} else {{
                        newsHtml += `<p>표시할 뉴스 기사가 없습니다.</p>`;
                    }}

                    contentDiv.innerHTML = newsHtml;

                }} else {{
                    contentDiv.innerHTML = `
                        <h4 style="color: #f44336;">❌ API 오류</h4>
                        <p>${{data.message}}</p>
                    `;
                }}
            }} catch (error) {{
                clearInterval(loadingInterval);
                contentDiv.innerHTML = `
                    <h4 style="color: #f44336;">❌ 네트워크 오류</h4>
                    <p>${{error.message}}</p>
                    <p>해결방법: 페이지를 새로고침(F5) 후 다시 시도하세요.</p>
                `;
            }} finally {{
                isRequesting = false;
                buttons.forEach(btn => btn.disabled = false);
            }}
        }}

        async function testHealthAPI() {{
            const resultDiv = document.getElementById('test-result');
            const contentDiv = document.getElementById('test-content');
            resultDiv.style.display = 'block';
            contentDiv.innerHTML = '⏳ 서버 상태 확인 중...';
            try {{
                const response = await fetch('/api/health');
                const data = await response.json();
                contentDiv.innerHTML = `
                    <h4 style="color: #4CAF50;">✅ 서버 상태 확인 완료!</h4>
                    <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px;">
                        <p><strong>상태:</strong> ${{data.status}}</p>
                        <p><strong>AI 모델:</strong> ${{data.ai_model}}</p>
                        <p><strong>AI 활성화:</strong> ${{data.ai_enabled ? '✅ Yes' : '❌ No'}}</p>
                        <p><strong>추천 뉴스:</strong> ${{data.recommendation_enabled ? '✅ 활성화' : '❌ 비활성화'}}</p>
                        <p><strong>확인 시간:</strong> ${{new Date(data.timestamp).toLocaleString('ko-KR')}}</p>
                    </div>
                `;
            }} catch (error) {{
                contentDiv.innerHTML = `
                    <h4 style="color: #f44336;">❌ 상태 확인 실패</h4>
                    <p>${{error.message}}</p>
                `;
            }}
        }}
        </script>
    </body>
    </html>
    """

@app.route('/api/news/<category>')
def get_news_api(category):
    """카테고리별 AI 고속 요약 뉴스 API + 추천 뉴스"""
    try:
        # 추천 뉴스 처리
        if category == 'recommended':
            print(f"🎯 API 요청: 오늘의 추천 뉴스")
            result = ai_app.get_recommended_news_fast(5)
            cache_status = "캐시" if result.get('cached', False) else "AI 처리"
            print(f"✅ API 응답 완료: 추천 뉴스 ({cache_status})")
            return jsonify(result)

        # 기존 카테고리 처리
        # 영문을 한글로 변환
        korean_category = CATEGORY_MAPPING.get(category, category)

        # 한글 카테고리도 지원 (하위 호환성)
        if category in SUPPORTED_CATEGORIES:
            korean_category = category

        if korean_category not in SUPPORTED_CATEGORIES:
            return jsonify({
                "status": "error",
                "message": f"지원하지 않는 카테고리입니다.",
                "supported_categories": list(CATEGORY_MAPPING.keys()),
                "data": []
            }), 400

        print(f"📡 API 요청: {korean_category} 뉴스 (영문: {category})")

        # 고속 요약 메서드 호출
        result = ai_app.get_dual_summarized_news_fast(korean_category, 5)

        cache_status = "캐시" if result.get('cached', False) else "AI 처리"
        print(f"✅ API 응답 완료: {korean_category} ({cache_status})")
        return jsonify(result)

    except Exception as e:
        print(f"❌ API 오류: {e}")
        return jsonify({
            "status": "error",
            "message": f"서버 오류: {str(e)}",
            "category": category,
            "data": [],
            "timestamp": datetime.now(KST).isoformat()
        }), 500

@app.route('/api/health')
def health_check():
    """서버 상태 확인 API"""
    try:
        ai_enabled_status = ai_app.summarizer.summarizer is not None
        current_kst_time = datetime.now(KST)

        return jsonify({
            "status": "healthy",
            "ai_model": "T5 Korean Summarization (Length Optimized)",
            "ai_enabled": ai_enabled_status,
            "summary_modes": ["preview", "detailed"],
            "supported_categories": list(CATEGORY_MAPPING.keys()),
            "recommendation_enabled": True,  # 추천 뉴스 기능 상태
            "performance": "고속 배치 처리 + 추천 뉴스",
            "cache_enabled": True,
            "timestamp": current_kst_time.isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(KST).isoformat()
        }), 500

@app.route('/api/test')
def test_endpoint():
    """간단한 연결 테스트 API"""
    return jsonify({
        "status": "success",
        "message": "길이 최적화 AI 뉴스 서버 + 추천 뉴스 기능이 정상 작동 중입니다!",
        "timestamp": datetime.now(KST).isoformat(),
        "endpoints": {
            "news": "/api/news/{category}",
            "recommended": "/api/news/recommended",  # 새로운 추천 엔드포인트
            "health": "/api/health",
            "test": "/api/test"
        },
        "categories": list(CATEGORY_MAPPING.keys()),
        "new_features": ["오늘의 추천 뉴스 (전체 장르 랜덤)"],
        "performance": "길이 최적화됨 + 추천 기능"
    })

# --- 서버 실행 부분 ---

def run_flask_server():
    """Flask 서버 실행"""
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

def start_server_with_ngrok():
    """Flask + ngrok 서버 시작"""
    print("🚀 Flask 서버를 백그라운드에서 시작합니다...")

    server_thread = threading.Thread(target=run_flask_server)
    server_thread.daemon = True
    server_thread.start()

    print("⏳ 서버 시작 및 AI 모델 로딩 대기 중...")
    time.sleep(10)

    try:
        # ngrok 토큰 라인을 주석 처리하거나 새 토큰으로 교체
        ngrok.set_auth_token("30KyKWx0zSS7ZJpm5TnJOKdI6fC_7y1Lta8QCMEyH6ZLjjrxj")

        public_url = ngrok.connect(8000).public_url
        print(f"✅ 서버가 성공적으로 시작되었습니다!")
        print(f"🌐 공개 URL: {public_url}")
        return str(public_url)
    except Exception as e:
        print(f"❌ ngrok 연결 실패: {e}")
        print("🔧 로컬에서만 접속 가능: http://localhost:8000")
        return "http://localhost:8000"

# 🌐 자동 서버 시작
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 AI 뉴스 길이 최적화 + 추천 뉴스 웹서버 시작")
    print("="*60)

    public_url = start_server_with_ngrok()

    print(f"""
🎉 AI 뉴스 웹서버 시작 완료!
📱 웹 브라우저에서 접속:   {public_url}

🔗 주요 API 엔드포인트:
    {public_url}/api/health           (서버 상태)
    {public_url}/api/news/recommended (🎯 오늘의 추천 뉴스 - NEW!)
    {public_url}/api/news/politics    (정치 뉴스)
    {public_url}/api/news/economy     (경제 뉴스)
    {public_url}/api/news/entertainment (연예 뉴스)
    {public_url}/api/news/sports      (스포츠 뉴스)
    {public_url}/api/news/society     (사회 뉴스)
    {public_url}/api/news/health      (건강 뉴스)
    {public_url}/api/news/lifestyle   (생활문화 뉴스)

⚡ 길이 최적화:
    ✅ 미리보기: 1줄 핵심 요약 (15-60자)
    ✅ 상세보기: 5문장 상세 요약 (200-400자)
    ✅ 배치 처리로 3-5배 속도 향상
    ✅ 캐싱으로 재요청 시 1-2초 응답

🎯 새로운 추천 뉴스 기능:
    ✅ 전체 7개 장르에서 랜덤하게 5개 기사 선택
    ✅ 1분마다 자동 갱신
    ✅ 다양한 주제의 뉴스를 한 번에!
    ✅ API: /api/news/recommended

    """)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 서버 중지됨")
        try:
            ngrok.kill()
        except:
            pass
