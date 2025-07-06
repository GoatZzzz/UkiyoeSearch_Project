import React, { useEffect, useState } from 'react'; import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CardActionArea,
  CircularProgress,
  Pagination,
  Button
} from '@mui/material';
import { useSearchParams, useNavigate } from 'react-router-dom';

const SearchResultsPage = () => {
  const [searchParams] = useSearchParams();
  const query = searchParams.get('query') || '';
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const itemsPerPage = 10;
  const navigate = useNavigate();

  useEffect(() => {
    const fetchResults = async () => {
      if (!query.trim()) return;
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8000/search', {
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
      setPage(1);
    };
    fetchResults();
  }, [query]);

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  const totalPages = Math.ceil(results.length / itemsPerPage);
  const startIndex = (page - 1) * itemsPerPage;
  const endIndex = page * itemsPerPage;
  const currentItems = results.slice(startIndex, endIndex);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4">搜索结果: {query}</Typography>
        <Button variant="outlined" onClick={() => navigate('/')}>
         Continue Search
        </Button>
      </Box>
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : results.length === 0 ? (
        <Typography variant="h6" align="center" sx={{ mt: 2 }}>
          Loading.....
        </Typography>
      ) : (
        <>
          <Grid container spacing={2}>
            {currentItems.map((item, index) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={`${item.rank}-${index}`}>
                <Card sx={{ borderRadius: '12px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                  <CardActionArea
                    component="a"
                    // 点击后跳转到指定链接，此处示例为固定链接，你可以根据 item 数据动态生成
                    href="https://ukiyo-e.org/image/jaodb/Hiroshige_1_Ando-53_Stations_of_the_Tokaido-Hara-00029064-020222-F06"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <CardMedia
                      component="img"
                      // 设定最大高度，并使用 objectFit:"contain" 使图片完整显示，不会被裁剪
                      sx={{
                        height: 'auto',
                        maxHeight: 300,
                        objectFit: 'contain'
                      }}
                      image={
                        item.image_url.startsWith('http')
                          ? item.image_url
                          : `http://127.0.0.1:8000${item.image_url}`
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
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
          <Box mt={4} display="flex" justifyContent="center">
            <Pagination count={totalPages} page={page} onChange={handlePageChange} color="primary" />
          </Box>
        </>
      )}
    </Container>
  );
};

export default SearchResultsPage;
