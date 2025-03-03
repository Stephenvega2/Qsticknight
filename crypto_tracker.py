import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional
import traceback

logger = logging.getLogger(__name__)

def scrape_coinmarketcap() -> List[Dict]:
    """Scrape cryptocurrency data from CoinMarketCap"""
    try:
        url = 'https://coinmarketcap.com/'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        crypto_data = []
        # Updated selector for the latest CoinMarketCap structure
        for row in soup.select('tbody tr')[:10]:
            try:
                name_element = row.select_one('td:nth-child(3) p')
                price_element = row.select_one('td:nth-child(4) span')
                change_24h_element = row.select_one('td:nth-child(5) span')

                if name_element and price_element and change_24h_element:
                    name = name_element.text.strip()
                    price = price_element.text.strip().replace('$', '').replace(',', '')
                    change_24h = change_24h_element.text.strip().replace('%', '')
                    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    crypto_data.append({
                        'name': name,
                        'price': float(price),
                        'change_24h': float(change_24h),
                        'last_updated': last_updated
                    })
                    logger.info(f"Successfully scraped {name} from CoinMarketCap")
            except Exception as e:
                logger.error(f"Error processing crypto row: {str(e)}")
                continue

        return crypto_data
    except Exception as e:
        logger.error(f"Error scraping CoinMarketCap: {str(e)}")
        return []

def scrape_cryptocom() -> List[Dict]:
    """Scrape cryptocurrency data from Crypto.com"""
    try:
        url = 'https://crypto.com/price'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        crypto_data = []
        # Updated selector for the latest Crypto.com structure
        for row in soup.select('tr[class*="css"]')[:10]:
            try:
                name_element = row.select_one('span[class*="name"]')
                price_element = row.select_one('div[class*="price"]')
                change_24h_element = row.select_one('td[class*="change"]')

                if name_element and price_element and change_24h_element:
                    name = name_element.text.strip()
                    price = price_element.text.strip().replace('$', '').replace(',', '')
                    change_24h = change_24h_element.text.strip().replace('%', '')
                    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    crypto_data.append({
                        'name': name,
                        'price': float(price),
                        'change_24h': float(change_24h),
                        'last_updated': last_updated
                    })
                    logger.info(f"Successfully scraped {name} from Crypto.com")
            except Exception as e:
                logger.error(f"Error processing crypto row: {str(e)}")
                continue

        return crypto_data
    except Exception as e:
        logger.error(f"Error scraping Crypto.com: {str(e)}")
        return []

def scrape_yahoo_finance() -> List[Dict]:
    """Scrape cryptocurrency data from Yahoo Finance"""
    try:
        url = 'https://finance.yahoo.com/cryptocurrencies'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        crypto_data = []
        for row in soup.select('table tbody tr')[:10]:
            try:
                name_element = row.select_one('td[data-field="name"] a')
                price_element = row.select_one('td[data-field="regularMarketPrice"]')
                change_24h_element = row.select_one('td[data-field="regularMarketChangePercent"]')

                if name_element and price_element and change_24h_element:
                    name = name_element.text.strip()
                    price = price_element.text.strip().replace(',', '')
                    change_24h = change_24h_element.text.strip().replace('%', '')
                    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    crypto_data.append({
                        'name': name,
                        'price': float(price),
                        'change_24h': float(change_24h),
                        'last_updated': last_updated
                    })
                    logger.info(f"Successfully scraped {name} from Yahoo Finance")
            except Exception as e:
                logger.error(f"Error processing crypto row: {str(e)}")
                continue

        return crypto_data
    except Exception as e:
        logger.error(f"Error scraping Yahoo Finance: {str(e)}")
        return []

def get_wallet_recommendations(crypto_name: str) -> Dict[str, str]:
    """Get wallet recommendations for specific cryptocurrencies with location restrictions"""
    wallet_recommendations = {
        'Bitcoin': {
            'hardware': {
                'name': 'Ledger Nano X',
                'description': 'Best overall hardware wallet with excellent security features',
                'usa_restricted': False,
                'alternatives': 'Available worldwide, no restrictions',
                'airdrop_support': False,
                'integrated_apps': ['Ledger Live']
            },
            'software': {
                'name': 'BlueWallet',
                'description': 'User-friendly mobile wallet with good security',
                'usa_restricted': False,
                'alternatives': 'For USA users, also consider Coinbase Wallet or Cash App',
                'airdrop_support': True,
                'integrated_apps': ['Lightning Network', 'Exchange', 'BTC Markets']
            },
            'web': {
                'name': 'Coinbase Wallet',
                'description': 'Popular web wallet with strong security measures',
                'usa_restricted': False,
                'alternatives': 'Fully available in USA with KYC requirements',
                'airdrop_support': True,
                'integrated_apps': ['Coinbase Exchange', 'NFT Marketplace', 'DeFi Browser']
            }
        },
        'Ethereum': {
            'hardware': {
                'name': 'Trezor Model T',
                'description': 'Premium hardware wallet with ETH support',
                'usa_restricted': False,
                'alternatives': 'Available worldwide, no restrictions',
                'airdrop_support': False,
                'integrated_apps': ['Trezor Suite']
            },
            'software': {
                'name': 'MetaMask',
                'description': 'Industry standard for ETH and ERC-20 tokens',
                'usa_restricted': False,
                'alternatives': 'Fully available in USA, some DApps may have restrictions',
                'airdrop_support': True,
                'integrated_apps': ['Token Swap', 'NFT Support', 'DApp Browser', 'Bridge']
            },
            'web': {
                'name': 'MyEtherWallet',
                'description': 'Established web interface for ETH transactions',
                'usa_restricted': False,
                'alternatives': 'Available in USA, some features require VPN',
                'airdrop_support': True,
                'integrated_apps': ['DeFi Integration', 'NFT Manager', 'MakerDAO']
            }
        },
        'BNB': {
            'hardware': {
                'name': 'SafePal S1',
                'description': 'Binance-backed hardware wallet',
                'usa_restricted': False,
                'alternatives': 'Available worldwide, consider Ledger for USA users',
                'airdrop_support': False,
                'integrated_apps': ['SafePal App']
            },
            'software': {
                'name': 'Trust Wallet',
                'description': 'Official Binance wallet with BNB support',
                'usa_restricted': False,
                'alternatives': 'USA users have limited access to Binance DEX features',
                'airdrop_support': True,
                'integrated_apps': ['DEX', 'DApp Browser', 'NFT Collection', 'Staking']
            },
            'web': {
                'name': 'Binance Chain Wallet',
                'description': 'Native web wallet for BNB chain',
                'usa_restricted': True,
                'alternatives': 'USA users should use MetaMask with BSC network instead',
                'airdrop_support': True,
                'integrated_apps': ['Binance DEX', 'LaunchPool', 'Yield Farming']
            }
        },
        'Solana': {
            'hardware': {
                'name': 'Ledger Nano S Plus',
                'description': 'Secure hardware wallet for SOL',
                'usa_restricted': False,
                'alternatives': 'Available worldwide, no restrictions',
                'airdrop_support': False,
                'integrated_apps': ['Ledger Live']
            },
            'software': {
                'name': 'Phantom',
                'description': 'Popular Solana wallet with great UX',
                'usa_restricted': False,
                'alternatives': 'Fully available in USA, some DeFi features restricted',
                'airdrop_support': True,
                'integrated_apps': ['Raydium', 'Magic Eden', 'Jupiter', 'Orca']
            },
            'web': {
                'name': 'Solflare',
                'description': 'Web wallet optimized for Solana ecosystem',
                'usa_restricted': False,
                'alternatives': 'Available in USA, some DEX features may be limited',
                'airdrop_support': True,
                'integrated_apps': ['Staking', 'NFT Gallery', 'Token Swap']
            }
        },
        'Cardano': {
            'hardware': {
                'name': 'Trezor Model One',
                'description': 'Affordable hardware wallet with ADA support',
                'usa_restricted': False,
                'alternatives': 'Available worldwide, no restrictions',
                'airdrop_support': False,
                'integrated_apps': ['Trezor Suite']
            },
            'software': {
                'name': 'Yoroi',
                'description': 'Official light wallet for Cardano',
                'usa_restricted': False,
                'alternatives': 'Fully available in USA with some DeFi restrictions',
                'airdrop_support': True,
                'integrated_apps': ['Staking Center', 'DApp Connector', 'NFT Gallery']
            },
            'web': {
                'name': 'Daedalus',
                'description': 'Full node wallet for advanced users',
                'usa_restricted': False,
                'alternatives': 'Available in USA, requires significant storage space',
                'airdrop_support': True,
                'integrated_apps': ['Stake Pool', 'Voting Center', 'Token Registry']
            }
        },
        'default': {
            'hardware': {
                'name': 'Ledger Nano X or Trezor Model T',
                'description': 'Universal hardware wallets',
                'usa_restricted': False,
                'alternatives': 'Both options fully available in USA',
                'airdrop_support': False,
                'integrated_apps': ['Basic Management']
            },
            'software': {
                'name': 'Trust Wallet',
                'description': 'Multi-currency software wallet',
                'usa_restricted': False,
                'alternatives': 'USA users may have limited access to some DeFi features',
                'airdrop_support': True,
                'integrated_apps': ['DApp Browser', 'Exchange', 'NFT Support']
            },
            'web': {
                'name': 'Coinbase Wallet',
                'description': 'Secure web wallet for multiple assets',
                'usa_restricted': False,
                'alternatives': 'Fully compliant with USA regulations',
                'airdrop_support': True,
                'integrated_apps': ['Exchange Integration', 'NFT Marketplace', 'DeFi Portal']
            }
        }
    }

    return wallet_recommendations.get(crypto_name, wallet_recommendations['default'])

def get_crypto_insights() -> Dict:
    """Get combined and analyzed crypto data with explanations and wallet recommendations"""
    try:
        # First try CoinMarketCap
        data_coinmarketcap = scrape_coinmarketcap()
        logger.info(f"CoinMarketCap returned {len(data_coinmarketcap)} entries")

        # If CoinMarketCap fails, try Crypto.com
        if not data_coinmarketcap:
            data_cryptocom = scrape_cryptocom()
            logger.info(f"Crypto.com returned {len(data_cryptocom)} entries")
            combined_data = data_cryptocom
        else:
            combined_data = data_coinmarketcap

        if not combined_data:
            logger.error("No cryptocurrency data could be retrieved from any source")
            return {
                'top_performers': [],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': 'Unable to retrieve cryptocurrency data from any source'
            }

        # Sort by 24h change percentage
        top_performers = sorted(combined_data, key=lambda x: x['change_24h'], reverse=True)[:5]

        # Add wallet recommendations and enhanced explanations
        for crypto in top_performers:
            wallet_info = get_wallet_recommendations(crypto['name'])
            crypto['wallet_recommendations'] = wallet_info
            crypto['explanation'] = (
                f"{crypto['name']} is currently priced at ${crypto['price']:.2f} with a 24-hour change "
                f"of {crypto['change_24h']:.2f}%. This {'positive' if crypto['change_24h'] > 0 else 'negative'} "
                f"change indicates a {'strong' if crypto['change_24h'] > 0 else 'weak'} recent performance. "
                f"Last updated on {crypto['last_updated']}."
            )

        return {
            'top_performers': top_performers,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error getting crypto insights: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'top_performers': [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e)
        }