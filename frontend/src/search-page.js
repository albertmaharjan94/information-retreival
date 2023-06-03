import React, { useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBookJournalWhills, faBookOpen, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Link } from 'react-router-dom';


const SearchPage = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [page, setPage] = useState(1);
    const [size, setSize] = useState(30);
    const [searchResults, setSearchResults] = useState([]);

    const getSearchResult = async () => {
        fetch(`http://localhost:8000/search?query=${searchQuery}&page=${page}&size=${size}`)
            .then(response => {
                return response.json();
            })
            .then(data => {
                setSearchResults(data);
            });
    };

    useEffect(() => {
        getSearchResult();
    }, [page, size]);

    const handleSearch = () => {
        getSearchResult();
    };

    useEffect(() => {
        setPage(searchResults.page);
        setSize(searchResults.size);
    }, [searchResults]);

    return (

        <div>
            <div className='headerBackground'>
                <img src='https://pureportal.coventry.ac.uk/skin/headerImage/' />
            </div>

            <div className='searchSection'>
                <div className='header'>
                    <div>
                        <FontAwesomeIcon icon={faBookJournalWhills} aria-hidden='true' style={{ marginRight: "20px", color: 'gray' }} />
                        <span>
                            Search for the publications (Coventry Univerity)
                        </span>
                        <FontAwesomeIcon icon={faBookJournalWhills} aria-hidden='true' style={{ marginLeft: "20px", color: 'gray' }} />
                    </div>
                </div>
                <div className='searchBox'>
                    <input
                        type="search"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Enter your search query"
                    />
                    <button onClick={handleSearch}>
                        <FontAwesomeIcon icon={faSearch} aria-hidden='true' />
                    </button>
                </div>
            </div>

            <div className='searchResult'>
                <div className='content'>
                    <h2>Search Results</h2>
                    <div className='pageDesc'>
                        {`${(page * size) - size} - ${page * size} out of ${searchResults.totalData}`}
                    </div>
                    {searchResults.results?.length > 0 ? (
                        <div className='card'>
                            {
                                searchResults.results?.map((value) => {
                                    return <div className='box' style={{ marginTop: "30px", marginBottom: "30px" }}>
                                        <div className='resultCard'>
                                            <a  target='_blank' href={value.pub_url}>
                                                <span>{value.name}</span>
                                            </a>
                                            <label className='cu_author'>{value.cu_author}</label>
                                            <label className='date'>{value.date}</label>
                                            <div style={{ marginTop: "5px", display:'flex', flexWrap:'wrap'}}>
                                                {
                                                    value.category.map(category => {
                                                        return <div className='category'
                                                            style={{ margin: '2px' }}
                                                        > {category}</div>;
                                                    })
                                                }
                                            </div>
                                        </div>
                                    </div>;
                                })
                            }
                        </div>
                    ) : (
                        <p>No search results found.</p>
                    )}


                    <div className='paginationDiv'>
                        <button className='previousBtn' onClick={() => {
                            if (page != 1) {
                                setPage(page - 1);
                                window.scrollTo({ top: 150, left: 0, behavior: 'smooth' });
                            }
                        }}>Prev</button>
                        <label>Page {page}</label>
                        <button className='nextBtn' onClick={() => {
                            setPage(page + 1);
                            window.scrollTo({ top: 150, left: 0, behavior: 'smooth' });
                        }
                        }>Next</button>
                    </div>

                </div>
            </div>


        </div>
    );
};

export default SearchPage;
