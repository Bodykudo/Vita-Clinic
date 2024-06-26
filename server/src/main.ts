import { NestFactory } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import * as basicAuth from 'express-basic-auth';

import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({
    origin: '*',
  });
  app.setGlobalPrefix('api');

  const swaggerPassword = app.get(ConfigService).get('SWAGGER_PASSWORD');
  app.use(
    ['/api/docs'],
    basicAuth({
      challenge: true,
      users: { admin: swaggerPassword },
    }),
  );

  const config = new DocumentBuilder()
    .setTitle('Vita Clinic API')
    .setDescription('Vita Clinic API Documentation')
    .setVersion('0.0')
    .build();
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  await app.listen(8000);
}
bootstrap();
