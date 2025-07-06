import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Box,
  Stack,
  IconButton
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import AddIcon from '@mui/icons-material/Add';

// 替换下面的 URL 为你实际的背景图片地址
const backgroundImages = [
  '/Mitsukuni_defying_the_skeleton_spectre_invoked_by_princess_Takiyasha.jpg',
  '/sc164646.jpg_fb1fb853-2acd-46ab-8435-1b4a55a683f4_sc164646.jpg',
  '/The_Great_Wave_off_Kanagawa.jpg'
];

const HomePage = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  // 用于控制当前背景图片的索引
  const [bgIndex, setBgIndex] = useState(0);

  // 每隔 5 秒切换背景图片
  useEffect(() => {
    const interval = setInterval(() => {
      setBgIndex((prevIndex) => (prevIndex + 1) % backgroundImages.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // 搜索处理，跳转到搜索结果页，同时将查询值作为 URL 参数传递
  const handleSearch = () => {
    if (query.trim()) {
      navigate(`/search?query=${encodeURIComponent(query)}`);
    }
  };

  // 按下回车时执行搜索
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      {/* 背景图片层：对每个图片进行绝对定位，并通过 opacity 渐变显示 */}
      {backgroundImages.map((img, index) => (
        <Box
          key={index}
          sx={{
            backgroundImage: `url(${img})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            opacity: index === bgIndex ? 1 : 0,
            transition: 'opacity 1s ease-in-out',
            zIndex: -1
          }}
        />
      ))}

      <Container
        maxWidth="sm"
        sx={{
          pt: 8,
          pb: 8,
          backgroundColor: 'rgba(255,255,255,0.85)',
          borderRadius: 2,
          mt: 4
        }}
      >
        {/* 标题 */}
        <Typography variant="h4" align="center" gutterBottom>
          What kind of Ukiyo-e would you like to find?
        </Typography>

        {/* 搜索区域容器 - 改为 position: relative */}
        <Box
          sx={{
            position: 'relative',
            mt: 2,
            maxWidth: '600px',
            margin: '0 auto'
          }}
        >
          {/* 搜索输入框 */}
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Enter the scene description."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            InputProps={{
              sx: {
                borderRadius: '9999px', // 圆润的边角
                boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                paddingLeft: '50px' // 左侧留出空间，避免文字与加号按钮重叠
              }
            }}
          />

          {/* 左下角加号按钮 */}
          <IconButton
            onClick={() => console.log('点击了加号')}
            sx={{
              position: 'absolute',
              left: 10,
              bottom: 10
            }}
          >
            <AddIcon />
          </IconButton>

          {/* 右下角搜索按钮 */}
          <Button
            variant="contained"
            onClick={handleSearch}
            sx={{
              position: 'absolute',
              right: 10,
              bottom: 6,
              borderRadius: '9999px',
              boxShadow: '0 2px 6px rgba(0,0,0,0.15)'
            }}
          >
            search
          </Button>
        </Box>

        {/* 下方仅保留“上传图片”和“更多...”按钮 */}
        <Stack
          direction="row"
          spacing={2}
          sx={{
            mt: 4,
            justifyContent: 'center'
          }}
        >
          <Button variant="outlined" onClick={() => console.log('上传图片')}>
            Upload pictures
          </Button>
          <Button variant="outlined" onClick={() => console.log('更多...')}>
            more...
          </Button>
        </Stack>
      </Container>
    </Box>
  );
};

export default HomePage;