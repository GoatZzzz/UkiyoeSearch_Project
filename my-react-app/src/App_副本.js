import React, { useState } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CircularProgress,
  Pagination,
  Box
} from '@mui/material';

const SearchResults = ({ results }) => {
  const itemsPerPage = 10;
  const [page, setPage] = useState(1);
  const totalPages = Math.ceil(results.length / itemsPerPage);

  const handleChange = (event, value) => {
    setPage(value);
  };

  const currentItems = results.slice((page - 1) * itemsPerPage, page * itemsPerPage);

  return (
    <Box>
      <Grid container spacing={2}>
        {currentItems.map((item) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={item.rank}>
            <Card>
              <CardMedia
                component="img"
                height="200"
                image={
                  item.image_url.startsWith('http')
                    ? item.image_url
                    : `http://localhost:8000${item.image_url}`
                }
                alt={item.photo_id}
              />
              <CardContent>
                <Typography variant="h6">Rank: {item.rank}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {item.photo_id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Distance: {item.distance.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Box mt={4} display="flex" justifyContent="center">
        <Pagination count={totalPages} page={page} onChange={handleChange} color="primary" />
      </Box>
    </Box>
  );
};

const SearchApp = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const response = await fetch('http://localhost::8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      } else {
        console.error('搜索请求失败');
      }
    } catch (error) {
      console.error('搜索时出错：', error);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h3" align="center" gutterBottom>
        Image Search
      </Typography>
      <Box sx={{ display: 'flex', mb: 4 }}>
        <TextField
          fullWidth
          label="请输入搜索内容"
          variant="outlined"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <Button variant="contained" color="primary" onClick={handleSearch} sx={{ ml: 2 }}>
          搜索
        </Button>
      </Box>
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : results.length > 0 ? (
        <SearchResults results={results} />
      ) : (
        <Typography variant="h6" align="center">
          暂无搜索结果
        </Typography>
      )}
    </Container>
  );
};

export default SearchApp;
